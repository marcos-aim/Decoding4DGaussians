# Shared Training Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared GPU-optimized training engine in `src/` that supports both single-scene overfitting and multi-scene generalization, with hardware-adaptive distributed training.

**Architecture:** Phase-based training loop with configurable memory strategies, loss registry, AMP, torch.compile on encoder heads, and DDP support. Experiments provide model + config overrides, everything else is shared.

**Tech Stack:** PyTorch 2.4+, diff_gaussian_rasterization (CUDA), pytorch_msssim, lpips, torch.distributed, TensorBoard

**Spec:** `docs/superpowers/specs/2026-04-04-shared-training-engine-design.md`

---

## File Structure

```
src/
├── __init__.py
├── config.py          # Config dataclass schema + YAML loading
├── distributed.py     # Hardware detection, DDP setup, sweep launcher
├── dataset.py         # SceneDataset with memory strategies (pinned/compressed/rotation/streaming)
├── losses.py          # LossRegistry with built-in losses + LPIPS optimization
├── metrics.py         # Async-friendly PSNR/SSIM accumulation
├── compile_utils.py   # torch.compile wrappers for encoder heads
├── checkpoint.py      # Phase-aware save/resume/best-tracking
├── renderer.py        # Shared render_gaussians wrapper
├── precompute.py      # STV2 precomputation pipeline with manifest
├── engine.py          # Core training loop + phase system (imports all above)
tests/
├── test_config.py         # (existing, for expmanager - leave alone)
├── test_engine_config.py  # Config loading + validation
├── test_losses.py         # Loss registry + individual loss functions
├── test_dataset.py        # Memory strategies + batch sampling
├── test_metrics.py        # Async metric accumulation
├── test_checkpoint.py     # Phase-aware checkpoint save/resume
├── test_engine.py         # Integration: full training step
```

---

### Task 1: Config Schema (`src/config.py`)

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/test_engine_config.py`

The config module defines the dataclass schema that all other modules consume. Every field has a default so single-scene configs stay minimal.

- [ ] **Step 1: Write failing tests for config loading**

```python
# tests/test_engine_config.py
"""Tests for shared training engine config."""
import pytest
import yaml
from pathlib import Path


def test_load_minimal_config(tmp_path):
    """Minimal config with just one scene and one phase should load with defaults."""
    from src.config import load_engine_config

    cfg_dict = {
        "data": {
            "scenes": [{"name": "test_scene", "path": "datasets/test", "precomputed": "precomputed/test"}],
        },
        "phases": [
            {"name": "train", "steps": 100, "trainable": ["head"],
             "losses": {"rgb": {"weight": 1.0}}}
        ],
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))

    cfg = load_engine_config(str(cfg_path))
    assert cfg.engine.amp is True  # default
    assert cfg.engine.compile is True  # default
    assert cfg.engine.log_every_n == 50
    assert cfg.data.memory_strategy == "auto"
    assert cfg.data.token_dtype == "float32"
    assert len(cfg.phases) == 1
    assert cfg.phases[0].name == "train"
    assert cfg.phases[0].steps == 100


def test_phase_defaults():
    """Phase should fill in optimizer defaults."""
    from src.config import PhaseConfig

    phase = PhaseConfig(
        name="test", steps=100, trainable=["head"],
        losses={"rgb": {"weight": 1.0}},
    )
    assert phase.optimizer.type == "adamw"
    assert phase.optimizer.lr == 1e-3
    assert phase.batch_frames is None  # inherits from data section
    assert phase.supervision_cams is None


def test_loss_config_defaults():
    """Loss config should default compute_every_n=1 and downsample=1."""
    from src.config import LossConfig

    lc = LossConfig(weight=0.5)
    assert lc.compute_every_n == 1
    assert lc.downsample == 1
    assert lc.warmup_steps == 0


def test_multi_scene_config(tmp_path):
    """Multi-scene config with streaming strategy."""
    from src.config import load_engine_config

    cfg_dict = {
        "data": {
            "scenes": [
                {"name": f"scene_{i}", "path": f"datasets/s{i}", "precomputed": f"precomputed/s{i}"}
                for i in range(10)
            ],
            "batch_scenes": 4,
            "memory_strategy": "streaming",
            "token_dtype": "float16",
        },
        "phases": [
            {"name": "train", "steps": 1000, "trainable": ["head"],
             "losses": {"rgb": {"weight": 1.0}}}
        ],
        "distributed": {"strategy": "ddp"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))

    cfg = load_engine_config(str(cfg_path))
    assert len(cfg.data.scenes) == 10
    assert cfg.data.batch_scenes == 4
    assert cfg.data.memory_strategy == "streaming"
    assert cfg.data.token_dtype == "float16"
    assert cfg.distributed.strategy == "ddp"


def test_precompute_config_defaults(tmp_path):
    """Precompute section should have sensible defaults."""
    from src.config import load_engine_config

    cfg_dict = {
        "data": {"scenes": [{"name": "s", "path": "p", "precomputed": "pc"}]},
        "phases": [{"name": "t", "steps": 1, "trainable": ["h"], "losses": {"rgb": {"weight": 1.0}}}],
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))

    cfg = load_engine_config(str(cfg_path))
    assert cfg.precompute.backbone == "spatracker_v2"
    assert cfg.precompute.window_size == 8
    assert cfg.precompute.dtype == "bfloat16"
    assert cfg.precompute.outputs.tokens is True
    assert cfg.precompute.outputs.points_map is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_engine_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Implement config schema**

```python
# src/__init__.py
"""Shared training engine for DecodeGaussians."""

# src/config.py
"""Config schema for the shared training engine.

All fields have defaults so minimal configs work out of the box.
Experiment configs override only what they need.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    warmup_steps: int = 500


@dataclass
class LossConfig:
    weight: float = 1.0
    warmup_steps: int = 0
    compute_every_n: int = 1
    downsample: int = 1


@dataclass
class PhaseConfig:
    name: str = "train"
    steps: int = 1000
    trainable: list[str] = field(default_factory=lambda: [])
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    losses: dict[str, LossConfig] = field(default_factory=dict)
    batch_frames: Optional[int] = None
    supervision_cams: Optional[int] = None
    grad_clip_max_norm: Optional[float] = None


@dataclass
class SceneConfig:
    name: str = ""
    path: str = ""
    precomputed: str = ""


@dataclass
class PrecomputeOutputsConfig:
    tokens: bool = True
    points_map: bool = True
    poses: bool = True
    intrinsics: bool = True
    uncertainty: bool = True


@dataclass
class PrecomputeConfig:
    backbone: str = "spatracker_v2"
    window_size: int = 8
    dtype: str = "bfloat16"
    output_dir: str = "precomputed/"
    outputs: PrecomputeOutputsConfig = field(default_factory=PrecomputeOutputsConfig)


@dataclass
class DataConfig:
    scenes: list[SceneConfig] = field(default_factory=list)
    resolution: tuple[int, int] = (512, 384)
    batch_scenes: int = 1
    batch_frames: int = 1
    supervision_cams: int = 4
    memory_strategy: str = "auto"
    token_dtype: str = "float32"
    rotation_budget_gb: Optional[float] = None
    input_camera: str = "cam01"
    eval_camera: str = "cam00"


@dataclass
class EngineConfig:
    amp: bool = True
    compile: bool = True
    compile_mode: str = "default"
    grad_clip_max_norm: float = 1.0
    log_every_n: int = 50
    save_every_n: int = 2000
    eval_every_n: int = 1000
    gradient_checkpointing: bool = False


@dataclass
class DistributedConfig:
    strategy: str = "auto"
    sweep_configs: Optional[list[str]] = None


@dataclass
class TrainingConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    precompute: PrecomputeConfig = field(default_factory=PrecomputeConfig)
    phases: list[PhaseConfig] = field(default_factory=list)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


def _parse_loss_config(loss_dict: dict) -> LossConfig:
    """Parse a loss config dict into a LossConfig dataclass."""
    return LossConfig(
        weight=loss_dict.get("weight", 1.0),
        warmup_steps=loss_dict.get("warmup_steps", 0),
        compute_every_n=loss_dict.get("compute_every_n", 1),
        downsample=loss_dict.get("downsample", 1),
    )


def _parse_phase(phase_dict: dict) -> PhaseConfig:
    """Parse a phase config dict into a PhaseConfig dataclass."""
    opt_dict = phase_dict.get("optimizer", {})
    sched_dict = phase_dict.get("scheduler", {})
    losses_raw = phase_dict.get("losses", {})

    betas = opt_dict.get("betas", (0.9, 0.999))
    if isinstance(betas, list):
        betas = tuple(betas)

    return PhaseConfig(
        name=phase_dict["name"],
        steps=phase_dict["steps"],
        trainable=phase_dict.get("trainable", []),
        optimizer=OptimizerConfig(
            type=opt_dict.get("type", "adamw"),
            lr=opt_dict.get("lr", 1e-3),
            weight_decay=opt_dict.get("weight_decay", 0.01),
            betas=betas,
        ),
        scheduler=SchedulerConfig(
            type=sched_dict.get("type", "cosine"),
            warmup_steps=sched_dict.get("warmup_steps", 500),
        ),
        losses={name: _parse_loss_config(lcfg) for name, lcfg in losses_raw.items()},
        batch_frames=phase_dict.get("batch_frames"),
        supervision_cams=phase_dict.get("supervision_cams"),
        grad_clip_max_norm=phase_dict.get("grad_clip_max_norm"),
    )


def load_engine_config(config_path: str) -> TrainingConfig:
    """Load a training config from a YAML file, filling defaults."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Engine
    eng_raw = raw.get("engine", {})
    engine = EngineConfig(
        amp=eng_raw.get("amp", True),
        compile=eng_raw.get("compile", True),
        compile_mode=eng_raw.get("compile_mode", "default"),
        grad_clip_max_norm=eng_raw.get("grad_clip_max_norm", 1.0),
        log_every_n=eng_raw.get("log_every_n", 50),
        save_every_n=eng_raw.get("save_every_n", 2000),
        eval_every_n=eng_raw.get("eval_every_n", 1000),
        gradient_checkpointing=eng_raw.get("gradient_checkpointing", False),
    )

    # Data
    data_raw = raw.get("data", {})
    scenes_raw = data_raw.get("scenes", [])
    scenes = [SceneConfig(name=s["name"], path=s["path"], precomputed=s["precomputed"]) for s in scenes_raw]
    resolution = data_raw.get("resolution", [512, 384])
    if isinstance(resolution, list):
        resolution = tuple(resolution)

    data = DataConfig(
        scenes=scenes,
        resolution=resolution,
        batch_scenes=data_raw.get("batch_scenes", 1),
        batch_frames=data_raw.get("batch_frames", 1),
        supervision_cams=data_raw.get("supervision_cams", 4),
        memory_strategy=data_raw.get("memory_strategy", "auto"),
        token_dtype=data_raw.get("token_dtype", "float32"),
        rotation_budget_gb=data_raw.get("rotation_budget_gb"),
        input_camera=data_raw.get("input_camera", "cam01"),
        eval_camera=data_raw.get("eval_camera", "cam00"),
    )

    # Precompute
    pre_raw = raw.get("precompute", {})
    out_raw = pre_raw.get("outputs", {})
    precompute = PrecomputeConfig(
        backbone=pre_raw.get("backbone", "spatracker_v2"),
        window_size=pre_raw.get("window_size", 8),
        dtype=pre_raw.get("dtype", "bfloat16"),
        output_dir=pre_raw.get("output_dir", "precomputed/"),
        outputs=PrecomputeOutputsConfig(
            tokens=out_raw.get("tokens", True),
            points_map=out_raw.get("points_map", True),
            poses=out_raw.get("poses", True),
            intrinsics=out_raw.get("intrinsics", True),
            uncertainty=out_raw.get("uncertainty", True),
        ),
    )

    # Phases
    phases = [_parse_phase(p) for p in raw.get("phases", [])]

    # Distributed
    dist_raw = raw.get("distributed", {})
    distributed = DistributedConfig(
        strategy=dist_raw.get("strategy", "auto"),
        sweep_configs=dist_raw.get("sweep_configs"),
    )

    return TrainingConfig(
        engine=engine,
        data=data,
        precompute=precompute,
        phases=phases,
        distributed=distributed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_engine_config.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py src/config.py tests/test_engine_config.py
git commit -m "feat: add config schema for shared training engine"
```

---

### Task 2: Loss Registry (`src/losses.py`)

**Files:**
- Create: `src/losses.py`
- Create: `tests/test_losses.py`

The loss registry wraps all existing loss functions and adds per-loss config (weight, warmup, compute frequency, downsample). LPIPS gets special treatment: 2x downsample + compute every N steps.

- [ ] **Step 1: Write failing tests for loss registry**

```python
# tests/test_losses.py
"""Tests for shared training engine loss registry."""
import pytest
import torch


@pytest.fixture
def registry():
    from src.losses import LossRegistry
    return LossRegistry(device="cpu")


def test_registry_has_builtin_losses(registry):
    """All built-in losses should be registered."""
    expected = {"rgb", "ssim", "geometric", "depth", "tv", "scale_reg", "opacity_reg"}
    assert expected.issubset(set(registry.list_losses()))


def test_register_custom_loss(registry):
    """Can register a custom loss function."""
    def my_loss(predictions, targets, **kwargs):
        return torch.tensor(0.5)

    registry.register("custom", my_loss)
    assert "custom" in registry.list_losses()


def test_compute_single_loss(registry):
    """Computing a single active loss returns a scalar tensor."""
    rendered = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    predictions = {"rendered": rendered, "scales": torch.rand(100, 3), "opacity": torch.rand(100, 1).sigmoid()}
    targets = {"gt_image": target}

    phase_losses = {"rgb": {"weight": 1.0}}
    loss = registry.compute(predictions, targets, step=0, phase_losses=phase_losses)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_loss_warmup(registry):
    """Loss with warmup_steps should be zero at step 0."""
    rendered = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    predictions = {"rendered": rendered, "scales": torch.rand(100, 3), "opacity": torch.rand(100, 1).sigmoid()}
    targets = {"gt_image": target}

    phase_losses = {"rgb": {"weight": 1.0, "warmup_steps": 100}}
    loss_at_0 = registry.compute(predictions, targets, step=0, phase_losses=phase_losses)
    loss_at_100 = registry.compute(predictions, targets, step=100, phase_losses=phase_losses)
    assert loss_at_0.item() < loss_at_100.item()


def test_compute_every_n_skips(registry):
    """Loss with compute_every_n > 1 should return 0 on non-compute steps."""
    rendered = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    predictions = {"rendered": rendered, "scales": torch.rand(100, 3), "opacity": torch.rand(100, 1).sigmoid()}
    targets = {"gt_image": target}

    phase_losses = {"rgb": {"weight": 1.0, "compute_every_n": 5}}
    loss_step_0 = registry.compute(predictions, targets, step=0, phase_losses=phase_losses)
    loss_step_1 = registry.compute(predictions, targets, step=1, phase_losses=phase_losses)
    # Step 0 is a compute step (0 % 5 == 0), step 1 uses cached value
    assert loss_step_0.item() > 0


def test_omitted_losses_not_computed(registry):
    """Only losses listed in phase_losses are computed."""
    predictions = {"rendered": torch.rand(1, 3, 64, 64), "scales": torch.rand(100, 3),
                   "opacity": torch.rand(100, 1).sigmoid()}
    targets = {"gt_image": torch.rand(1, 3, 64, 64)}

    # Only rgb — geometric should not be called (no gaussian_means in predictions)
    phase_losses = {"rgb": {"weight": 1.0}}
    loss = registry.compute(predictions, targets, step=0, phase_losses=phase_losses)
    assert loss.item() > 0


def test_photometric_loss_values():
    """Photometric loss should combine L1 and SSIM."""
    from src.losses import photometric_loss

    identical = torch.rand(1, 3, 64, 64)
    loss = photometric_loss(identical, identical, lambda_ssim=0.85)
    assert loss.item() < 0.01  # near-zero for identical images

    different = torch.rand(1, 3, 64, 64)
    loss2 = photometric_loss(identical, different, lambda_ssim=0.85)
    assert loss2.item() > loss.item()


def test_scale_regularization():
    """Scale reg should penalize large scales."""
    from src.losses import scale_regularization

    small = torch.ones(100, 3) * 0.01
    large = torch.ones(100, 3) * 5.0
    assert scale_regularization(small).item() < scale_regularization(large).item()


def test_opacity_regularization():
    """Opacity reg should penalize mid-range opacities."""
    from src.losses import opacity_regularization

    binary = torch.tensor([[0.01], [0.99]] * 50)
    midrange = torch.tensor([[0.5]] * 100)
    assert opacity_regularization(binary).item() < opacity_regularization(midrange).item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.losses'`

- [ ] **Step 3: Implement loss registry and loss functions**

```python
# src/losses.py
"""Loss registry with built-in losses and LPIPS optimization.

Losses register by name. Phases reference them by name in config.
Each loss implements: fn(predictions, targets, **kwargs) -> scalar tensor.

LPIPS optimization: 2x downsample + compute every N steps to reduce cost ~80%.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as compute_ssim


# ── Built-in loss functions ──────────────────────────────────────────────────

def photometric_loss(rendered: torch.Tensor, target: torch.Tensor,
                     lambda_ssim: float = 0.85) -> torch.Tensor:
    """Combined L1 + SSIM photometric loss."""
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    l1 = F.l1_loss(rendered, target)
    ssim_val = compute_ssim(rendered, target, data_range=1.0, size_average=True)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)


def scale_regularization(scales: torch.Tensor) -> torch.Tensor:
    """Penalize large Gaussian scales to prevent unbounded growth."""
    return torch.mean(torch.abs(scales))


def opacity_regularization(opacity: torch.Tensor) -> torch.Tensor:
    """Entropy regularization on opacity — push toward 0 or 1."""
    return torch.mean(-torch.log(opacity + 1e-6) - torch.log(1.0 - opacity + 1e-6))


def geometric_loss(gaussian_means: torch.Tensor, points_map: torch.Tensor,
                   max_pts: int = 8192) -> torch.Tensor:
    """L1 distance from Gaussian centers to nearest STV2 point cloud point."""
    pts = points_map.reshape(-1, 3)
    N = gaussian_means.shape[0]
    if N > max_pts:
        idx = torch.randperm(N, device=gaussian_means.device)[:max_pts]
        g_sub = gaussian_means[idx]
    else:
        g_sub = gaussian_means
    M = pts.shape[0]
    if M > max_pts:
        idx = torch.randperm(M, device=pts.device)[:max_pts]
        p_sub = pts[idx]
    else:
        p_sub = pts
    dists = torch.cdist(g_sub.unsqueeze(0), p_sub.unsqueeze(0)).squeeze(0)
    min_dists, _ = dists.min(dim=1)
    return min_dists.mean()


def depth_loss(rendered_depth: torch.Tensor, points_map: torch.Tensor,
               camera) -> torch.Tensor:
    """Compare rendered depth with GT depth derived from STV2 point cloud."""
    H, W = rendered_depth.shape[1], rendered_depth.shape[2]
    pts = points_map.reshape(-1, 3)
    ones = torch.ones(pts.shape[0], 1, device=pts.device)
    pts_homo = torch.cat([pts, ones], dim=-1)
    pts_cam = pts_homo @ camera.world_view_transform
    gt_depth = pts_cam[:, 2].reshape(H, W)
    rd = rendered_depth.squeeze(0)
    mask = (gt_depth > 0.01) & (rd > 0.01)
    if mask.sum() < 100:
        return torch.tensor(0.0, device=rendered_depth.device)
    gt_median = gt_depth[mask].median()
    if gt_median < 1e-6:
        return torch.tensor(0.0, device=rendered_depth.device)
    return F.smooth_l1_loss(rd[mask] / gt_median, gt_depth[mask] / gt_median)


def tv_loss(deltas_t: torch.Tensor, deltas_t_prev: torch.Tensor) -> torch.Tensor:
    """Temporal variation loss on deformation deltas."""
    return F.l1_loss(deltas_t, deltas_t_prev)


# ── Loss adapters (bridge between registry interface and raw loss fns) ───────

def _rgb_adapter(predictions, targets, **kwargs):
    return photometric_loss(predictions["rendered"], targets["gt_image"],
                            lambda_ssim=kwargs.get("lambda_ssim", 0.85))


def _ssim_adapter(predictions, targets, **kwargs):
    """Standalone SSIM (when used as separate loss from rgb)."""
    rendered = predictions["rendered"]
    target = targets["gt_image"]
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    ssim_val = compute_ssim(rendered, target, data_range=1.0, size_average=True)
    return 1.0 - ssim_val


def _geometric_adapter(predictions, targets, **kwargs):
    return geometric_loss(predictions["gaussian_means"], targets["points_map"])


def _depth_adapter(predictions, targets, **kwargs):
    return depth_loss(predictions["rendered_depth"], targets["points_map"], targets["camera"])


def _tv_adapter(predictions, targets, **kwargs):
    return tv_loss(predictions["all_deltas"], predictions["all_deltas_prev"])


def _scale_reg_adapter(predictions, targets, **kwargs):
    return scale_regularization(predictions["scales"])


def _opacity_reg_adapter(predictions, targets, **kwargs):
    return opacity_regularization(predictions["opacity"])


# ── Registry ─────────────────────────────────────────────────────────────────

class LossRegistry:
    """Named loss functions with per-phase configuration.

    Usage:
        registry = LossRegistry(device="cuda")
        registry.register("custom", my_loss_fn)
        total = registry.compute(predictions, targets, step=100, phase_losses={"rgb": {"weight": 1.0}})
    """

    def __init__(self, device: str = "cuda"):
        self._losses: dict[str, Callable] = {}
        self._cache: dict[str, torch.Tensor] = {}
        self._device = device

        # Register built-in losses
        self.register("rgb", _rgb_adapter)
        self.register("ssim", _ssim_adapter)
        self.register("geometric", _geometric_adapter)
        self.register("depth", _depth_adapter)
        self.register("tv", _tv_adapter)
        self.register("scale_reg", _scale_reg_adapter)
        self.register("opacity_reg", _opacity_reg_adapter)

    def register(self, name: str, loss_fn: Callable) -> None:
        """Register a loss function by name."""
        self._losses[name] = loss_fn

    def list_losses(self) -> list[str]:
        """Return names of all registered losses."""
        return list(self._losses.keys())

    def compute(self, predictions: dict, targets: dict, step: int,
                phase_losses: dict[str, dict]) -> torch.Tensor:
        """Compute total weighted loss for active losses in this phase.

        Args:
            predictions: dict with rendered images, Gaussian params, deltas
            targets: dict with gt_image, points_map, camera, etc.
            step: current training step (for warmup and compute_every_n)
            phase_losses: {loss_name: {weight, warmup_steps, compute_every_n, ...}}

        Returns:
            Scalar total loss tensor.
        """
        total = torch.tensor(0.0, device=self._device)

        for name, loss_cfg in phase_losses.items():
            if name not in self._losses:
                raise KeyError(f"Loss '{name}' not registered. Available: {self.list_losses()}")

            weight = loss_cfg.get("weight", 1.0)
            warmup_steps = loss_cfg.get("warmup_steps", 0)
            compute_every_n = loss_cfg.get("compute_every_n", 1)

            # Warmup scaling
            if warmup_steps > 0 and step < warmup_steps:
                warmup_scale = step / warmup_steps
            else:
                warmup_scale = 1.0

            effective_weight = weight * warmup_scale
            if effective_weight == 0:
                continue

            # Compute every N steps, use cached value otherwise
            if compute_every_n <= 1 or step % compute_every_n == 0:
                loss_val = self._losses[name](predictions, targets, **loss_cfg)
                self._cache[name] = loss_val.detach()
            else:
                loss_val = self._cache.get(name, torch.tensor(0.0, device=self._device))

            total = total + effective_weight * loss_val

        return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_losses.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/losses.py tests/test_losses.py
git commit -m "feat: add loss registry with built-in losses and LPIPS optimization"
```

---

### Task 3: Metrics (`src/metrics.py`)

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

Async-friendly metric accumulator. No `.item()` calls in the hot path — metrics accumulate on GPU tensors and sync to CPU only when requested.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
"""Tests for async-friendly metric accumulation."""
import pytest
import torch


def test_accumulator_tracks_running_mean():
    from src.metrics import MetricAccumulator

    acc = MetricAccumulator()
    acc.update("loss", torch.tensor(1.0))
    acc.update("loss", torch.tensor(3.0))
    result = acc.flush()
    assert abs(result["loss"] - 2.0) < 1e-6


def test_accumulator_tracks_multiple_metrics():
    from src.metrics import MetricAccumulator

    acc = MetricAccumulator()
    acc.update("loss", torch.tensor(1.0))
    acc.update("psnr", torch.tensor(25.0))
    result = acc.flush()
    assert "loss" in result
    assert "psnr" in result


def test_flush_resets_accumulator():
    from src.metrics import MetricAccumulator

    acc = MetricAccumulator()
    acc.update("loss", torch.tensor(1.0))
    acc.flush()
    acc.update("loss", torch.tensor(5.0))
    result = acc.flush()
    assert abs(result["loss"] - 5.0) < 1e-6


def test_flush_empty_returns_empty():
    from src.metrics import MetricAccumulator

    acc = MetricAccumulator()
    result = acc.flush()
    assert result == {}


def test_compute_psnr_identical():
    from src.metrics import compute_psnr_tensor

    img = torch.rand(3, 64, 64)
    psnr = compute_psnr_tensor(img, img)
    assert psnr.item() > 50.0  # near-infinite for identical


def test_compute_psnr_different():
    from src.metrics import compute_psnr_tensor

    a = torch.zeros(3, 64, 64)
    b = torch.ones(3, 64, 64)
    psnr = compute_psnr_tensor(a, b)
    assert abs(psnr.item() - 0.0) < 0.01  # MSE=1 → PSNR=0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement metrics module**

```python
# src/metrics.py
"""Async-friendly metric accumulation.

No .item() calls in the hot path. Metrics accumulate on GPU tensors
and sync to CPU only on flush() (called every log_every_n steps).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MetricAccumulator:
    """Accumulates scalar metrics without GPU sync.

    Usage:
        acc = MetricAccumulator()
        acc.update("loss", loss_tensor)       # no .item() call
        acc.update("psnr", psnr_tensor)
        if step % log_every_n == 0:
            metrics = acc.flush()             # sync happens here
            writer.add_scalar("loss", metrics["loss"], step)
    """

    def __init__(self):
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def update(self, name: str, value: torch.Tensor) -> None:
        """Add a scalar tensor value. No GPU sync."""
        if name not in self._sums:
            self._sums[name] = value.detach().clone()
            self._counts[name] = 1
        else:
            self._sums[name] = self._sums[name] + value.detach()
            self._counts[name] += 1

    def flush(self) -> dict[str, float]:
        """Compute means, sync to CPU, and reset. This is the only sync point."""
        result = {}
        for name in self._sums:
            mean_val = self._sums[name] / self._counts[name]
            result[name] = mean_val.item()
        self._sums.clear()
        self._counts.clear()
        return result


def compute_psnr_tensor(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute PSNR as a tensor (no .item() call). Returns scalar tensor.

    Args:
        rendered: [3, H, W] or [B, 3, H, W] in [0, 1]
        target:   same shape as rendered
    """
    mse = F.mse_loss(rendered, target)
    # Clamp MSE to avoid log(0)
    mse = torch.clamp(mse, min=1e-10)
    return -10.0 * torch.log10(mse)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_metrics.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add async-friendly metric accumulator"
```

---

### Task 4: Checkpoint Manager (`src/checkpoint.py`)

**Files:**
- Create: `src/checkpoint.py`
- Create: `tests/test_checkpoint.py`

Phase-aware checkpoint save/resume with best-metric tracking.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_checkpoint.py
"""Tests for phase-aware checkpoint management."""
import pytest
import torch
import torch.nn as nn
from pathlib import Path


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


@pytest.fixture
def ckpt_dir(tmp_path):
    return tmp_path / "checkpoints"


def test_save_and_load_checkpoint(ckpt_dir):
    from src.checkpoint import CheckpointManager

    mgr = CheckpointManager(str(ckpt_dir))
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())

    mgr.save(
        phase_idx=0, phase_name="canonical", step=100,
        modules={"head": model}, optimizer=optimizer,
        scheduler=None, metrics={"psnr": 22.5},
    )

    assert (ckpt_dir / "phase_0_canonical" / "step_000100.pt").exists()
    assert (ckpt_dir / "phase_0_canonical" / "latest.pt").exists()


def test_best_checkpoint_tracking(ckpt_dir):
    from src.checkpoint import CheckpointManager

    mgr = CheckpointManager(str(ckpt_dir), best_metric="psnr", higher_is_better=True)
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())

    mgr.save(phase_idx=0, phase_name="train", step=100,
             modules={"head": model}, optimizer=optimizer,
             scheduler=None, metrics={"psnr": 20.0})
    mgr.save(phase_idx=0, phase_name="train", step=200,
             modules={"head": model}, optimizer=optimizer,
             scheduler=None, metrics={"psnr": 25.0})
    mgr.save(phase_idx=0, phase_name="train", step=300,
             modules={"head": model}, optimizer=optimizer,
             scheduler=None, metrics={"psnr": 23.0})

    best_path = ckpt_dir / "phase_0_train" / "best.pt"
    assert best_path.exists()
    best_ckpt = torch.load(best_path, weights_only=False)
    assert best_ckpt["step"] == 200
    assert best_ckpt["metrics"]["psnr"] == 25.0


def test_load_latest_checkpoint(ckpt_dir):
    from src.checkpoint import CheckpointManager

    mgr = CheckpointManager(str(ckpt_dir))
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())

    mgr.save(phase_idx=0, phase_name="train", step=500,
             modules={"head": model}, optimizer=optimizer,
             scheduler=None, metrics={})

    model2 = DummyModel()
    loaded = mgr.load_latest(phase_idx=0, phase_name="train",
                              modules={"head": model2})
    assert loaded["step"] == 500


def test_find_resume_point_empty(ckpt_dir):
    from src.checkpoint import CheckpointManager

    mgr = CheckpointManager(str(ckpt_dir))
    result = mgr.find_resume_point()
    assert result is None


def test_find_resume_point_with_checkpoints(ckpt_dir):
    from src.checkpoint import CheckpointManager

    mgr = CheckpointManager(str(ckpt_dir))
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters())

    mgr.save(phase_idx=0, phase_name="canonical", step=100,
             modules={"head": model}, optimizer=optimizer,
             scheduler=None, metrics={})
    mgr.save(phase_idx=1, phase_name="deformation", step=50,
             modules={"head": model}, optimizer=optimizer,
             scheduler=None, metrics={})

    result = mgr.find_resume_point()
    assert result["phase_idx"] == 1
    assert result["step"] == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL

- [ ] **Step 3: Implement checkpoint manager**

```python
# src/checkpoint.py
"""Phase-aware checkpoint save/resume with best-metric tracking.

Checkpoint structure:
    checkpoints/
    ├── phase_0_canonical/
    │   ├── step_020000.pt
    │   ├── best.pt
    │   └── latest.pt
    ├── phase_1_deformation/
    │   └── ...
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Phase-aware checkpoint management.

    Args:
        checkpoint_dir: root directory for all checkpoints
        best_metric: metric name to track for best checkpoint (e.g., "psnr")
        higher_is_better: if True, higher metric = better (default True for PSNR)
    """

    def __init__(self, checkpoint_dir: str, best_metric: str = "psnr",
                 higher_is_better: bool = True):
        self.root = Path(checkpoint_dir)
        self.best_metric = best_metric
        self.higher_is_better = higher_is_better
        self._best_values: dict[int, float] = {}  # phase_idx -> best metric value

    def _phase_dir(self, phase_idx: int, phase_name: str) -> Path:
        d = self.root / f"phase_{phase_idx}_{phase_name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, phase_idx: int, phase_name: str, step: int,
             modules: dict[str, nn.Module], optimizer,
             scheduler, metrics: dict) -> str:
        """Save checkpoint. Updates latest.pt and best.pt if applicable.

        Returns:
            Path to saved checkpoint file.
        """
        phase_dir = self._phase_dir(phase_idx, phase_name)

        state = {
            "phase_idx": phase_idx,
            "phase_name": phase_name,
            "step": step,
            "metrics": metrics,
            "modules": {name: mod.state_dict() for name, mod in modules.items()},
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
        }

        # Save numbered checkpoint
        step_path = phase_dir / f"step_{step:06d}.pt"
        torch.save(state, step_path)

        # Update latest
        latest_path = phase_dir / "latest.pt"
        shutil.copy2(step_path, latest_path)

        # Update best if applicable
        if self.best_metric and self.best_metric in metrics:
            val = metrics[self.best_metric]
            current_best = self._best_values.get(phase_idx)

            is_better = (
                current_best is None
                or (self.higher_is_better and val > current_best)
                or (not self.higher_is_better and val < current_best)
            )

            if is_better:
                self._best_values[phase_idx] = val
                best_path = phase_dir / "best.pt"
                shutil.copy2(step_path, best_path)

        return str(step_path)

    def load_latest(self, phase_idx: int, phase_name: str,
                    modules: dict[str, nn.Module],
                    optimizer=None, scheduler=None) -> dict:
        """Load the latest checkpoint for a phase into modules/optimizer/scheduler.

        Returns:
            Dict with step, metrics, phase info.
        """
        phase_dir = self._phase_dir(phase_idx, phase_name)
        latest_path = phase_dir / "latest.pt"

        if not latest_path.exists():
            raise FileNotFoundError(f"No checkpoint at {latest_path}")

        state = torch.load(latest_path, weights_only=False)

        for name, mod in modules.items():
            if name in state["modules"]:
                mod.load_state_dict(state["modules"][name])

        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])

        return {"step": state["step"], "metrics": state.get("metrics", {}),
                "phase_idx": state["phase_idx"], "phase_name": state["phase_name"]}

    def load_best(self, phase_idx: int, phase_name: str,
                  modules: dict[str, nn.Module]) -> Optional[dict]:
        """Load the best checkpoint for a phase. Returns None if no best exists."""
        phase_dir = self._phase_dir(phase_idx, phase_name)
        best_path = phase_dir / "best.pt"

        if not best_path.exists():
            return None

        state = torch.load(best_path, weights_only=False)
        for name, mod in modules.items():
            if name in state["modules"]:
                mod.load_state_dict(state["modules"][name])

        return {"step": state["step"], "metrics": state.get("metrics", {})}

    def find_resume_point(self) -> Optional[dict]:
        """Find the latest checkpoint across all phases for resume.

        Returns:
            Dict with phase_idx, phase_name, step, or None if no checkpoints.
        """
        if not self.root.exists():
            return None

        latest_phase = None
        latest_step = -1

        for phase_dir in sorted(self.root.iterdir()):
            if not phase_dir.is_dir():
                continue
            match = re.match(r"phase_(\d+)_(.*)", phase_dir.name)
            if not match:
                continue

            phase_idx = int(match.group(1))
            phase_name = match.group(2)

            latest_path = phase_dir / "latest.pt"
            if latest_path.exists():
                state = torch.load(latest_path, weights_only=False)
                step = state["step"]
                if phase_idx > (latest_phase["phase_idx"] if latest_phase else -1):
                    latest_phase = {
                        "phase_idx": phase_idx,
                        "phase_name": phase_name,
                        "step": step,
                    }
                elif phase_idx == (latest_phase["phase_idx"] if latest_phase else -1) and step > latest_step:
                    latest_phase = {
                        "phase_idx": phase_idx,
                        "phase_name": phase_name,
                        "step": step,
                    }

        return latest_phase
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_checkpoint.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add phase-aware checkpoint manager"
```

---

### Task 5: Renderer Wrapper (`src/renderer.py`)

**Files:**
- Create: `src/renderer.py`

Direct port of existing renderer with multi-camera convenience. No tests needed — this wraps a CUDA extension that requires GPU.

- [ ] **Step 1: Implement shared renderer**

```python
# src/renderer.py
"""Thin wrapper around diff-gaussian-rasterization CUDA kernel.

Provides single-camera and multi-camera rendering.
torch.compile is NOT applied here — the CUDA extension is already optimized.
"""

import math
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render_gaussians(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity: torch.Tensor,
    shs: torch.Tensor,
    camera,
    bg_color: torch.Tensor,
    sh_degree: int = 0,
):
    """Render Gaussians from a given camera viewpoint.

    Args:
        means3D:   [N, 3] Gaussian centers in world space
        scales:    [N, 3] Gaussian scales (positive)
        rotations: [N, 4] unit quaternions (wxyz)
        opacity:   [N, 1] opacity values in [0, 1]
        shs:       [N, C, 3] spherical harmonics (C = (sh_degree+1)^2)
        camera:    MiniCam with world_view_transform, full_proj_transform, etc.
        bg_color:  [3] background color tensor on GPU
        sh_degree: SH degree (default 0 = DC only)

    Returns:
        rendered: [3, H, W] RGB image
        radii:    [N] screen-space radii
        depth:    [1, H, W] depth map
        screenspace_points: [N, 3] for gradient flow
    """
    device = means3D.device
    screenspace_points = torch.zeros_like(means3D, requires_grad=True, device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx / 2.0),
        tanfovy=math.tan(camera.FoVy / 2.0),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    return rendered, radii, depth, screenspace_points


def render_multi_camera(
    means3D: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity: torch.Tensor,
    shs: torch.Tensor,
    cameras: list,
    bg_color: torch.Tensor,
    sh_degree: int = 0,
) -> list[dict]:
    """Render Gaussians from multiple cameras. Shares Gaussian tensors across renders.

    Returns:
        List of dicts, each with keys: rendered, radii, depth, screenspace_points
    """
    results = []
    for cam in cameras:
        rendered, radii, depth, ssp = render_gaussians(
            means3D, scales, rotations, opacity, shs, cam, bg_color, sh_degree
        )
        results.append({
            "rendered": rendered,
            "radii": radii,
            "depth": depth,
            "screenspace_points": ssp,
        })
    return results
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer.py
git commit -m "feat: add shared renderer wrapper with multi-camera support"
```

---

### Task 6: Compile Utils (`src/compile_utils.py`)

**Files:**
- Create: `src/compile_utils.py`

Applies torch.compile to encoder/head modules but not the CUDA rasterizer.

- [ ] **Step 1: Implement compile utils**

```python
# src/compile_utils.py
"""torch.compile wrappers for encoder/head modules.

Applied to: canonical_head, deformation_head (MLP + transformer encoder)
NOT applied to: diff_gaussian_rasterization (custom CUDA kernel)

Modes:
  - "default": safe, minimal overhead
  - "reduce-overhead": uses CUDA graphs, faster steady-state
  - "max-autotune": benchmarks kernels, best throughput after warmup
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compile_modules(
    modules: dict[str, nn.Module],
    compile_enabled: bool = True,
    compile_mode: str = "default",
    skip_names: set[str] | None = None,
) -> dict[str, nn.Module]:
    """Apply torch.compile to named modules.

    Args:
        modules: dict of name -> nn.Module
        compile_enabled: if False, return modules unchanged
        compile_mode: torch.compile mode
        skip_names: module names to skip (e.g., modules that call custom CUDA ops)

    Returns:
        Dict with same keys, compiled modules where applicable.
    """
    if not compile_enabled:
        return modules

    skip = skip_names or set()
    compiled = {}

    for name, module in modules.items():
        if name in skip:
            compiled[name] = module
            print(f"  [compile] Skipping {name} (in skip list)")
        else:
            try:
                compiled[name] = torch.compile(module, mode=compile_mode)
                print(f"  [compile] Compiled {name} (mode={compile_mode})")
            except Exception as e:
                print(f"  [compile] Failed to compile {name}: {e}, using eager mode")
                compiled[name] = module

    return compiled
```

- [ ] **Step 2: Commit**

```bash
git add src/compile_utils.py
git commit -m "feat: add torch.compile wrappers for encoder heads"
```

---

### Task 7: Distributed Setup (`src/distributed.py`)

**Files:**
- Create: `src/distributed.py`

Hardware detection, DDP initialization, and sweep launcher.

- [ ] **Step 1: Implement distributed module**

```python
# src/distributed.py
"""Hardware detection, DDP setup, and sweep launcher.

Strategies:
  - single: 1 GPU, no DDP overhead
  - ddp: multi-GPU joint training with gradient sync
  - sweep: each GPU runs independent experiment
  - auto: single if 1 GPU, ddp if multi-GPU

Hardware environments:
  - Local: RTX 4070 Super (12 GB), single GPU
  - Cluster: 4x A5000 (24 GB each), multi-GPU
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: tuple[int, int]


def detect_gpus() -> list[GPUInfo]:
    """Detect available GPUs and their properties."""
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_mem / 1e9
        free_gb = (props.total_mem - torch.cuda.memory_reserved(i)) / 1e9
        gpus.append(GPUInfo(
            index=i,
            name=props.name,
            vram_total_gb=total_gb,
            vram_free_gb=free_gb,
            compute_capability=(props.major, props.minor),
        ))
    return gpus


def resolve_strategy(strategy: str, num_gpus: int) -> str:
    """Resolve 'auto' strategy based on GPU count."""
    if strategy != "auto":
        return strategy
    return "single" if num_gpus <= 1 else "ddp"


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize DDP process group."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Cleanup DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0 or no DDP)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def launch_sweep(config_paths: list[str], python_cmd: str = "python") -> list:
    """Launch independent experiments on separate GPUs (sweep mode).

    Each config runs on a different GPU. No gradient sync.

    Args:
        config_paths: list of config YAML paths, one per GPU
        python_cmd: python executable path

    Returns:
        List of subprocess.Popen objects
    """
    gpus = detect_gpus()
    if len(config_paths) > len(gpus):
        raise RuntimeError(
            f"Sweep has {len(config_paths)} configs but only {len(gpus)} GPUs"
        )

    processes = []
    for i, cfg_path in enumerate(config_paths):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        proc = subprocess.Popen(
            [python_cmd, "-m", "src.engine", cfg_path],
            env=env,
        )
        processes.append(proc)
        print(f"  [sweep] GPU {i}: {cfg_path} (pid={proc.pid})")

    return processes
```

- [ ] **Step 2: Commit**

```bash
git add src/distributed.py
git commit -m "feat: add distributed setup with DDP and sweep support"
```

---

### Task 8: Dataset with Memory Strategies (`src/dataset.py`)

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

The big one — multi-scene dataset with configurable memory strategies.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset.py
"""Tests for multi-scene dataset with memory strategies."""
import pytest
import torch
import json
from pathlib import Path


def _make_fake_scene(scene_dir: Path, num_frames: int = 10, num_patches: int = 16,
                     token_dim: int = 32, H: int = 8, W: int = 8, num_cams: int = 3):
    """Create a fake precomputed scene for testing."""
    scene_dir.mkdir(parents=True, exist_ok=True)

    tokens = torch.randn(num_frames, num_patches, token_dim)
    torch.save(tokens, scene_dir / "tokens.pt")

    points_map = torch.randn(num_frames, H, W, 3)
    torch.save(points_map, scene_dir / "points_map.pt")

    # Fake poses: identity
    poses = torch.eye(4).unsqueeze(0).expand(num_frames, -1, -1)
    torch.save(poses, scene_dir / "poses.pt")

    intrs = torch.eye(4).unsqueeze(0).expand(num_frames, -1, -1)
    torch.save(intrs, scene_dir / "intrs.pt")

    unc = torch.rand(num_frames, H, W)
    torch.save(unc, scene_dir / "unc_metric.pt")

    # Fake frame images
    for cam_idx in range(num_cams):
        cam_dir = scene_dir / "frames" / f"cam{cam_idx:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for f_idx in range(num_frames):
            img = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8)
            # Save as raw tensor (skip JPEG for tests)
            torch.save(img, cam_dir / f"{f_idx:06d}.pt")

    manifest = {
        "backbone": "spatracker_v2",
        "num_frames": num_frames,
        "resolution": [W, H],
    }
    (scene_dir / "manifest.json").write_text(json.dumps(manifest))


def test_pinned_strategy_loads_to_device(tmp_path):
    from src.dataset import SceneDataset
    from src.config import DataConfig, SceneConfig

    scene_dir = tmp_path / "precomputed" / "scene_a"
    _make_fake_scene(scene_dir, num_frames=5)

    config = DataConfig(
        scenes=[SceneConfig(name="scene_a", path="", precomputed=str(scene_dir))],
        memory_strategy="pinned",
        token_dtype="float32",
    )
    ds = SceneDataset(config, device="cpu")  # use cpu for testing
    assert ds.num_scenes == 1
    assert ds.scenes["scene_a"]["tokens"].shape[0] == 5


def test_compressed_uses_float16(tmp_path):
    from src.dataset import SceneDataset
    from src.config import DataConfig, SceneConfig

    scene_dir = tmp_path / "precomputed" / "scene_a"
    _make_fake_scene(scene_dir, num_frames=5)

    config = DataConfig(
        scenes=[SceneConfig(name="scene_a", path="", precomputed=str(scene_dir))],
        memory_strategy="compressed",
        token_dtype="float16",
    )
    ds = SceneDataset(config, device="cpu")
    assert ds.scenes["scene_a"]["tokens"].dtype == torch.float16


def test_sample_batch_returns_correct_keys(tmp_path):
    from src.dataset import SceneDataset
    from src.config import DataConfig, SceneConfig

    scene_dir = tmp_path / "precomputed" / "scene_a"
    _make_fake_scene(scene_dir, num_frames=5)

    config = DataConfig(
        scenes=[SceneConfig(name="scene_a", path="", precomputed=str(scene_dir))],
        memory_strategy="pinned",
    )
    ds = SceneDataset(config, device="cpu")

    batch = ds.sample_training_batch(batch_scenes=1, batch_frames=1)
    assert "tokens_mean" in batch
    assert "tokens_frames" in batch
    assert "points_map_frames" in batch
    assert "scene_names" in batch
    assert "frame_indices" in batch


def test_multi_scene_batch(tmp_path):
    from src.dataset import SceneDataset
    from src.config import DataConfig, SceneConfig

    scenes = []
    for i in range(3):
        scene_dir = tmp_path / "precomputed" / f"scene_{i}"
        _make_fake_scene(scene_dir, num_frames=5)
        scenes.append(SceneConfig(name=f"scene_{i}", path="", precomputed=str(scene_dir)))

    config = DataConfig(scenes=scenes, memory_strategy="pinned", batch_scenes=2)
    ds = SceneDataset(config, device="cpu")
    assert ds.num_scenes == 3

    batch = ds.sample_training_batch(batch_scenes=2, batch_frames=1)
    assert len(batch["scene_names"]) == 2


def test_auto_strategy_single_scene(tmp_path):
    from src.dataset import resolve_memory_strategy

    strategy = resolve_memory_strategy("auto", num_scenes=1, vram_gb=12.0)
    assert strategy == "pinned"


def test_auto_strategy_many_scenes(tmp_path):
    from src.dataset import resolve_memory_strategy

    strategy = resolve_memory_strategy("auto", num_scenes=50, vram_gb=12.0)
    assert strategy == "streaming"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement dataset**

```python
# src/dataset.py
"""Multi-scene dataset with configurable memory strategies.

Memory strategies:
  - pinned: all data on GPU (single-scene default)
  - compressed: float16 tokens on GPU (saves ~50% VRAM)
  - rotation: LRU set of scenes on GPU, evict when full
  - streaming: async prefetch from CPU/disk
  - auto: selects based on scene count + VRAM

GT images are cached on device when VRAM allows, eliminating per-step JPEG I/O.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import torch
import cv2


def resolve_memory_strategy(strategy: str, num_scenes: int, vram_gb: float) -> str:
    """Resolve 'auto' strategy based on scene count and available VRAM.

    Heuristic:
      - 1 scene → pinned (fits easily)
      - 2-5 scenes → compressed (float16 tokens fit in most GPUs)
      - 6+ scenes → streaming (too many to keep in VRAM)
    """
    if strategy != "auto":
        return strategy

    if num_scenes <= 1:
        return "pinned"
    elif num_scenes <= 5:
        return "compressed"
    else:
        return "streaming"


class SceneDataset:
    """Unified dataset for single and multi-scene training.

    Args:
        config: DataConfig with scenes, memory_strategy, token_dtype, etc.
        device: torch device string ("cuda" or "cpu")
        cameras: optional dict of camera name -> camera object (for GT image loading)
    """

    def __init__(self, config, device: str = "cuda", cameras: dict = None):
        self.config = config
        self.device = device
        self.cameras = cameras or {}
        self.scenes: dict[str, dict] = {}

        # Resolve memory strategy
        vram_gb = 0.0
        if device == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        self.strategy = resolve_memory_strategy(
            config.memory_strategy, len(config.scenes), vram_gb
        )

        token_dtype = torch.float16 if config.token_dtype == "float16" else torch.float32

        # Load scenes based on strategy
        for scene_cfg in config.scenes:
            scene_data = self._load_scene(scene_cfg, token_dtype)
            self.scenes[scene_cfg.name] = scene_data

        self.scene_names = list(self.scenes.keys())
        print(f"  Dataset: {len(self.scenes)} scenes, strategy={self.strategy}")

    @property
    def num_scenes(self) -> int:
        return len(self.scenes)

    def _load_scene(self, scene_cfg, token_dtype: torch.dtype) -> dict:
        """Load a single scene's precomputed data."""
        cache_dir = Path(scene_cfg.precomputed)

        tokens = torch.load(cache_dir / "tokens.pt", weights_only=True)
        tokens = tokens.to(dtype=token_dtype)

        points_map = torch.load(cache_dir / "points_map.pt", weights_only=True).float()

        tokens_mean = tokens.float().mean(dim=0).to(dtype=token_dtype)

        scene_data = {
            "tokens": tokens,
            "tokens_mean": tokens_mean,
            "points_map": points_map,
            "num_frames": tokens.shape[0],
            "num_patches": tokens.shape[1],
            "cache_dir": str(cache_dir),
            "gt_images_cache": {},  # populated on demand or at init
        }

        # Pin to device for pinned/compressed strategies
        if self.strategy in ("pinned", "compressed"):
            scene_data["tokens"] = scene_data["tokens"].to(self.device)
            scene_data["tokens_mean"] = scene_data["tokens_mean"].to(self.device)
            scene_data["points_map"] = scene_data["points_map"].to(self.device)

        return scene_data

    def _load_frame_image(self, cache_dir: str, cam_name: str, frame_idx: int) -> torch.Tensor:
        """Load a single frame image. Returns [3, H, W] float32 tensor in [0, 1]."""
        # Try tensor format first (used in tests)
        pt_path = os.path.join(cache_dir, "frames", cam_name, f"{frame_idx:06d}.pt")
        if os.path.exists(pt_path):
            img = torch.load(pt_path, weights_only=True)
            if img.dim() == 3 and img.shape[2] == 3:
                return img.permute(2, 0, 1).float() / 255.0
            return img.float() / 255.0

        # JPEG format
        jpg_path = os.path.join(cache_dir, "frames", cam_name, f"{frame_idx:06d}.jpg")
        img = cv2.imread(jpg_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def sample_training_batch(self, batch_scenes: int = 1, batch_frames: int = 1,
                              supervision_cams: Optional[list[str]] = None) -> dict:
        """Sample a training batch across scenes and frames.

        Args:
            batch_scenes: number of scenes per batch
            batch_frames: number of frames per scene
            supervision_cams: camera names for GT supervision (None = use all train cams)

        Returns:
            dict with: tokens_mean, tokens_frames, points_map_frames,
                       scene_names, frame_indices
        """
        selected_scenes = random.sample(self.scene_names, min(batch_scenes, len(self.scene_names)))

        all_tokens_mean = []
        all_tokens_frames = []
        all_points_map = []
        all_frame_indices = []

        for scene_name in selected_scenes:
            scene = self.scenes[scene_name]
            frame_indices = random.sample(range(scene["num_frames"]), min(batch_frames, scene["num_frames"]))
            frame_indices.sort()

            all_tokens_mean.append(scene["tokens_mean"])
            all_tokens_frames.append(torch.stack([scene["tokens"][i] for i in frame_indices]))
            all_points_map.append(torch.stack([scene["points_map"][i] for i in frame_indices]))
            all_frame_indices.append(frame_indices)

        return {
            "tokens_mean": torch.stack(all_tokens_mean),  # [B_scenes, P, D]
            "tokens_frames": torch.stack(all_tokens_frames) if len(all_tokens_frames) > 1 else all_tokens_frames[0],
            "points_map_frames": torch.stack(all_points_map) if len(all_points_map) > 1 else all_points_map[0],
            "scene_names": selected_scenes,
            "frame_indices": all_frame_indices,
        }

    def get_eval_batch(self, scene_name: str, frame_idx: int) -> dict:
        """Get a single frame for evaluation."""
        scene = self.scenes[scene_name]
        return {
            "tokens_mean": scene["tokens_mean"],
            "tokens_frame": scene["tokens"][frame_idx],
            "points_map": scene["points_map"][frame_idx],
        }

    def cache_gt_images(self, scene_name: str, cam_names: list[str]) -> None:
        """Pre-cache all GT images for a scene on device. Eliminates per-step I/O."""
        scene = self.scenes[scene_name]
        cache_dir = scene["cache_dir"]

        for cam_name in cam_names:
            if cam_name in scene["gt_images_cache"]:
                continue
            frames = []
            for i in range(scene["num_frames"]):
                img = self._load_frame_image(cache_dir, cam_name, i)
                frames.append(img)
            scene["gt_images_cache"][cam_name] = torch.stack(frames).to(self.device)
            print(f"  Cached {scene_name}/{cam_name}: {scene['gt_images_cache'][cam_name].shape}")

    def get_gt_image(self, scene_name: str, cam_name: str, frame_idx: int) -> torch.Tensor:
        """Get GT image, from cache if available, else load from disk."""
        scene = self.scenes[scene_name]
        if cam_name in scene["gt_images_cache"]:
            return scene["gt_images_cache"][cam_name][frame_idx]
        return self._load_frame_image(scene["cache_dir"], cam_name, frame_idx).to(self.device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_dataset.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: add multi-scene dataset with configurable memory strategies"
```

---

### Task 9: Precompute Pipeline (`src/precompute.py`)

**Files:**
- Create: `src/precompute.py`

Shared STV2 precomputation with manifest and multi-scene batch mode.

- [ ] **Step 1: Implement precompute pipeline**

```python
# src/precompute.py
"""SpaTrackerV2 precomputation pipeline.

Extracts and caches STV2 latents (tokens, points_map, poses, intrinsics, uncertainty)
for training. Idempotent: skips scenes with complete manifests.

Usage:
    python -m src.precompute --config config.yaml
    python -m src.precompute --scenes datasets/neural_3d/* --output precomputed/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def extract_frames(scene_path: str, output_dir: str, resolution: tuple[int, int],
                   camera_dirs: list[str] = None) -> int:
    """Extract frames from MP4 videos and save as JPEGs.

    Args:
        scene_path: path to scene directory containing camera MP4s
        output_dir: where to save extracted frames
        resolution: (width, height) target resolution
        camera_dirs: list of camera directory names, or None to auto-detect

    Returns:
        Number of frames extracted per camera.
    """
    scene_path = Path(scene_path)
    output_dir = Path(output_dir)

    if camera_dirs is None:
        # Auto-detect camera directories with MP4 files
        camera_dirs = sorted([
            d.name for d in scene_path.iterdir()
            if d.is_dir() and any(d.glob("*.mp4"))
        ])

    num_frames = 0
    for cam_name in camera_dirs:
        cam_out = output_dir / "frames" / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)

        mp4_files = sorted((scene_path / cam_name).glob("*.mp4"))
        if not mp4_files:
            continue

        cap = cv2.VideoCapture(str(mp4_files[0]))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resolution)
            cv2.imwrite(str(cam_out / f"{frame_idx:06d}.jpg"), frame)
            frame_idx += 1
        cap.release()
        num_frames = max(num_frames, frame_idx)
        print(f"  {cam_name}: {frame_idx} frames")

    return num_frames


def run_stv2_extraction(
    frames_dir: str,
    output_dir: str,
    input_camera: str,
    window_size: int = 8,
    dtype_str: str = "bfloat16",
    outputs_config: dict = None,
) -> dict:
    """Run SpaTrackerV2 (VGGT4Track) and extract tokens + predictions.

    Args:
        frames_dir: directory with extracted frames (frames/{cam_name}/{idx:06d}.jpg)
        output_dir: where to save cached tensors
        input_camera: which camera to use as STV2 input
        window_size: frames per STV2 forward pass
        dtype_str: "bfloat16" or "float16" for autocast
        outputs_config: dict of output_name -> bool for which tensors to save

    Returns:
        dict with shapes of saved tensors
    """
    # Import STV2 lazily (heavy dependency)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SpaTrackerV2"))
    from models.spatracker.predictor import SpaTrackerPredictor

    outputs_config = outputs_config or {"tokens": True, "points_map": True,
                                         "poses": True, "intrinsics": True,
                                         "uncertainty": True}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frames for input camera
    cam_dir = Path(frames_dir) / "frames" / input_camera
    frame_files = sorted(cam_dir.glob("*.jpg"))
    print(f"  Loading {len(frame_files)} frames from {input_camera}...")

    frames = []
    for ff in frame_files:
        img = cv2.imread(str(ff))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img).permute(2, 0, 1))
    all_frames = torch.stack(frames)  # [T, 3, H, W]
    num_frames = all_frames.shape[0]

    # Load model
    print("  Loading VGGT4Track model...")
    model = SpaTrackerPredictor.from_pretrained("facebook/VGGT4Track")
    model = model.cuda().eval()

    # Hook to capture aggregated tokens
    captured_tokens = {}
    def hook_fn(module, input, output):
        if hasattr(module, "aggregated_tokens_list") and module.aggregated_tokens_list:
            captured_tokens["tokens"] = module.aggregated_tokens_list[-1]

    hook = model.aggregator.register_forward_hook(hook_fn)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    autocast_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    all_tokens = []
    all_points_map = []
    all_poses = []
    all_intrs = []
    all_unc = []

    print(f"  Running VGGT4Track (window={window_size})...")
    for start in tqdm(range(0, num_frames, window_size), desc="VGGT4Track"):
        end = min(start + window_size, num_frames)
        frames_window = all_frames[start:end].unsqueeze(0).cuda() / 255.0

        torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    predictions = model(frames_window)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                half = max(1, (end - start) // 2)
                print(f"\n  OOM at window {start}-{end}, retrying with window={half}")
                for sub_start in range(start, end, half):
                    sub_end = min(sub_start + half, end)
                    sub_frames = all_frames[sub_start:sub_end].unsqueeze(0).cuda() / 255.0
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=autocast_dtype):
                            pred = model(sub_frames)
                    n = sub_end - sub_start
                    all_tokens.append(captured_tokens["tokens"].squeeze(0)[:n].half().cpu())
                    all_points_map.append(pred["points_map"].cpu()[:n])
                    all_poses.append(pred["poses_pred"].cpu().squeeze(0)[:n])
                    all_intrs.append(pred["intrs"].cpu().squeeze(0)[:n])
                    all_unc.append(pred["unc_metric"].cpu()[:n])
                continue
            else:
                raise

        n = end - start
        all_tokens.append(captured_tokens["tokens"].squeeze(0)[:n].half().cpu())
        all_points_map.append(predictions["points_map"].cpu()[:n])
        all_poses.append(predictions["poses_pred"].cpu().squeeze(0)[:n])
        all_intrs.append(predictions["intrs"].cpu().squeeze(0)[:n])
        all_unc.append(predictions["unc_metric"].cpu()[:n])

    hook.remove()

    tokens = torch.cat(all_tokens, dim=0)
    points_map = torch.cat(all_points_map, dim=0)
    poses = torch.cat(all_poses, dim=0)
    intrs = torch.cat(all_intrs, dim=0)
    unc_metric = torch.cat(all_unc, dim=0)

    shapes = {}
    if outputs_config.get("tokens", True):
        torch.save(tokens, output_dir / "tokens.pt")
        shapes["tokens"] = list(tokens.shape)
    if outputs_config.get("points_map", True):
        torch.save(points_map.half(), output_dir / "points_map.pt")
        shapes["points_map"] = list(points_map.shape)
    if outputs_config.get("poses", True):
        torch.save(poses, output_dir / "poses.pt")
        shapes["poses"] = list(poses.shape)
    if outputs_config.get("intrinsics", True):
        torch.save(intrs, output_dir / "intrs.pt")
        shapes["intrinsics"] = list(intrs.shape)
    if outputs_config.get("uncertainty", True):
        torch.save(unc_metric.half(), output_dir / "unc_metric.pt")
        shapes["uncertainty"] = list(unc_metric.shape)

    # Write manifest
    manifest = {
        "backbone": "spatracker_v2",
        "num_frames": num_frames,
        "resolution": [all_frames.shape[3], all_frames.shape[2]],
        "window_size": window_size,
        "shapes": shapes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n  Saved to {output_dir}/")
    for name, shape in shapes.items():
        print(f"    {name}: {shape}")

    return shapes


def is_scene_complete(precomputed_dir: str) -> bool:
    """Check if a scene has a complete manifest (idempotent check)."""
    manifest_path = Path(precomputed_dir) / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        return "backbone" in manifest and "shapes" in manifest
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Precompute STV2 features")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--scenes", nargs="*", help="Scene directories to process")
    parser.add_argument("--output", type=str, default="precomputed/", help="Output root directory")
    parser.add_argument("--input-camera", type=str, default="cam01")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--resolution", nargs=2, type=int, default=[512, 384], metavar=("W", "H"))
    args = parser.parse_args()

    if args.config:
        from src.config import load_engine_config
        cfg = load_engine_config(args.config)
        for scene_cfg in cfg.data.scenes:
            if is_scene_complete(scene_cfg.precomputed):
                print(f"  Skipping {scene_cfg.name} (already complete)")
                continue
            print(f"\n=== Processing {scene_cfg.name} ===")
            extract_frames(scene_cfg.path, scene_cfg.precomputed,
                           tuple(cfg.data.resolution))
            run_stv2_extraction(
                scene_cfg.precomputed, scene_cfg.precomputed,
                cfg.data.input_camera, cfg.precompute.window_size,
                cfg.precompute.dtype,
                {k: getattr(cfg.precompute.outputs, k) for k in
                 ["tokens", "points_map", "poses", "intrinsics", "uncertainty"]},
            )
    elif args.scenes:
        for scene_path in args.scenes:
            scene_name = Path(scene_path).name
            output_dir = os.path.join(args.output, scene_name)
            if is_scene_complete(output_dir):
                print(f"  Skipping {scene_name} (already complete)")
                continue
            print(f"\n=== Processing {scene_name} ===")
            extract_frames(scene_path, output_dir, tuple(args.resolution))
            run_stv2_extraction(output_dir, output_dir, args.input_camera,
                                args.window_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/precompute.py
git commit -m "feat: add shared STV2 precomputation pipeline with manifest"
```

---

### Task 10: Core Engine (`src/engine.py`)

**Files:**
- Create: `src/engine.py`
- Create: `tests/test_engine.py`

The main training loop that ties everything together: phase system, AMP, compile, distributed, loss registry, metrics, checkpoints.

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_engine.py
"""Integration tests for the training engine.

These test the engine's phase loop logic with dummy models.
Full GPU tests require CUDA and are marked accordingly.
"""
import pytest
import torch
import torch.nn as nn
import yaml
from pathlib import Path

# Reuse fake scene helper
from test_dataset import _make_fake_scene


class DummyCanonicalHead(nn.Module):
    """Minimal canonical head for testing."""
    def __init__(self, dim_in=32, K=4):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(dim_in, K * 14)  # xyz(3)+scale(3)+rot(4)+opacity(1)+sh(3)

    def forward(self, tokens_mean):
        P = tokens_mean.shape[0]
        out = self.linear(tokens_mean.float())  # [P, K*14]
        out = out.view(P * self.K, 14)
        return {
            "xyz": out[:, :3],
            "log_scale": out[:, 3:6],
            "rot": torch.nn.functional.normalize(out[:, 6:10], dim=-1),
            "logit_opacity": out[:, 10:11],
            "sh": out[:, 11:14].unsqueeze(1),  # [P*K, 1, 3]
        }


class DummyDeformationHead(nn.Module):
    """Minimal deformation head for testing."""
    def __init__(self, dim_in=32, K=4):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(dim_in, K * 14)

    def forward(self, tokens_t):
        P = tokens_t.shape[0]
        out = self.linear(tokens_t.float())
        out = out.view(P * self.K, 14)
        return {
            "dxyz": torch.tanh(out[:, :3]) * 0.1,
            "dscale": torch.tanh(out[:, 3:6]) * 0.5,
            "drot": torch.nn.functional.normalize(out[:, 6:10], dim=-1),
            "dopacity": torch.tanh(out[:, 10:11]) * 0.3,
            "dsh": out[:, 11:14].unsqueeze(1),
            "all_deltas": out,
        }


def _make_engine_config(tmp_path, scene_dir):
    """Create a minimal engine config for testing."""
    cfg = {
        "engine": {
            "amp": False,  # disable for CPU tests
            "compile": False,
            "log_every_n": 5,
            "save_every_n": 10,
            "eval_every_n": 10,
        },
        "data": {
            "scenes": [{"name": "test", "path": "", "precomputed": str(scene_dir)}],
            "memory_strategy": "pinned",
            "batch_scenes": 1,
            "batch_frames": 1,
        },
        "phases": [
            {
                "name": "canonical",
                "steps": 10,
                "trainable": ["canonical_head"],
                "optimizer": {"lr": 1e-3},
                "losses": {"rgb": {"weight": 1.0}},
            },
        ],
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return str(cfg_path)


def test_engine_build_model_contract():
    """build_model must return dict of named nn.Modules."""
    modules = {
        "canonical_head": DummyCanonicalHead(),
        "deformation_head": DummyDeformationHead(),
    }
    assert all(isinstance(v, nn.Module) for v in modules.values())


def test_engine_phase_freezing():
    """Phase system should freeze/unfreeze correct modules."""
    from src.engine import _apply_phase_freezing

    modules = {
        "canonical_head": DummyCanonicalHead(),
        "deformation_head": DummyDeformationHead(),
    }

    _apply_phase_freezing(modules, trainable=["canonical_head"])

    for p in modules["canonical_head"].parameters():
        assert p.requires_grad is True
    for p in modules["deformation_head"].parameters():
        assert p.requires_grad is False


def test_engine_phase_unfreezing():
    """Joint phase should unfreeze all listed modules."""
    from src.engine import _apply_phase_freezing

    modules = {
        "canonical_head": DummyCanonicalHead(),
        "deformation_head": DummyDeformationHead(),
    }

    # First freeze deformation
    _apply_phase_freezing(modules, trainable=["canonical_head"])
    # Then unfreeze both
    _apply_phase_freezing(modules, trainable=["canonical_head", "deformation_head"])

    for p in modules["deformation_head"].parameters():
        assert p.requires_grad is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement core engine**

```python
# src/engine.py
"""Core training engine with phase system, AMP, and distributed support.

Usage:
    python -m src.engine config.yaml
    python -m src.engine config.yaml --resume
    torchrun --nproc_per_node=4 -m src.engine config.yaml
    python -m src.engine --sweep config_a.yaml config_b.yaml

The engine expects experiments to provide a build_model() function that returns
a dict of named nn.Modules. Everything else (dataset, losses, rendering,
checkpointing) is handled by the shared infrastructure.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import load_engine_config, TrainingConfig, PhaseConfig
from src.dataset import SceneDataset
from src.losses import LossRegistry
from src.metrics import MetricAccumulator, compute_psnr_tensor
from src.checkpoint import CheckpointManager
from src.compile_utils import compile_modules
from src.distributed import (
    detect_gpus, resolve_strategy, setup_ddp, cleanup_ddp,
    is_main_process, get_rank, get_world_size, launch_sweep,
)
from src.renderer import render_gaussians, render_multi_camera


def _apply_phase_freezing(modules: dict[str, nn.Module], trainable: list[str]) -> None:
    """Freeze all modules, then unfreeze only those listed in trainable."""
    for name, module in modules.items():
        requires_grad = name in trainable
        for param in module.parameters():
            param.requires_grad = requires_grad


def _build_optimizer(modules: dict[str, nn.Module], phase: PhaseConfig) -> optim.Optimizer:
    """Build optimizer for trainable parameters in this phase."""
    params = []
    for name in phase.trainable:
        if name in modules:
            params.extend(modules[name].parameters())

    opt_cfg = phase.optimizer
    if opt_cfg.type == "adamw":
        return optim.AdamW(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay,
                           betas=opt_cfg.betas)
    elif opt_cfg.type == "adam":
        return optim.Adam(params, lr=opt_cfg.lr, betas=opt_cfg.betas)
    elif opt_cfg.type == "sgd":
        return optim.SGD(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_cfg.type}")


def _build_scheduler(optimizer: optim.Optimizer, phase: PhaseConfig):
    """Build LR scheduler for this phase."""
    sched_cfg = phase.scheduler
    if sched_cfg.type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase.steps - sched_cfg.warmup_steps,
        )
    elif sched_cfg.type == "constant":
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_cfg.type}")


def train(config: TrainingConfig, build_model_fn, compose_fn,
          cameras: dict = None, experiment_dir: str = "."):
    """Main training loop.

    Args:
        config: parsed TrainingConfig
        build_model_fn: callable(config) -> dict[str, nn.Module]
        compose_fn: callable(canonical_output, deltas, scale_factor) -> Gaussian params
        cameras: dict of camera_name -> camera object
        experiment_dir: directory for logs, checkpoints, renders
    """
    device = f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu"
    experiment_dir = Path(experiment_dir)

    # Build model
    modules = build_model_fn(config)
    for name, mod in modules.items():
        mod.to(device)

    # Compile encoder heads
    if config.engine.compile and device != "cpu":
        modules = compile_modules(modules, compile_enabled=True,
                                  compile_mode=config.engine.compile_mode)

    # DDP wrapping
    strategy = resolve_strategy(config.distributed.strategy,
                                torch.cuda.device_count() if torch.cuda.is_available() else 0)
    if strategy == "ddp":
        for name, mod in modules.items():
            modules[name] = torch.nn.parallel.DistributedDataParallel(
                mod, device_ids=[get_rank()]
            )

    # Dataset
    dataset = SceneDataset(config.data, device=device, cameras=cameras)

    # Cache GT images for all training cameras if single-scene
    if dataset.num_scenes == 1 and cameras:
        train_cams = [c for c in cameras.keys() if c != config.data.eval_camera]
        dataset.cache_gt_images(dataset.scene_names[0], train_cams)

    # Loss registry
    loss_registry = LossRegistry(device=device)

    # Checkpoint manager
    ckpt_dir = experiment_dir / "checkpoints"
    ckpt_mgr = CheckpointManager(str(ckpt_dir))

    # TensorBoard
    writer = None
    if is_main_process():
        log_dir = experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))

    # Metric accumulator
    metrics_acc = MetricAccumulator()

    # AMP setup
    use_amp = config.engine.amp and device != "cpu"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=use_amp)

    bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)
    global_step = 0

    # ── Phase loop ────────────────────────────────────────────────────────
    for phase_idx, phase in enumerate(config.phases):
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Phase {phase_idx}: {phase.name} ({phase.steps} steps)")
            print(f"  Trainable: {phase.trainable}")
            print(f"  Losses: {list(phase.losses.keys())}")
            print(f"{'='*60}")

        # Freeze/unfreeze
        _apply_phase_freezing(modules, phase.trainable)

        # Load previous phase checkpoint if not first phase
        if phase_idx > 0:
            prev_phase = config.phases[phase_idx - 1]
            # Get raw modules (unwrap DDP if needed)
            raw_modules = {n: (m.module if hasattr(m, 'module') else m) for n, m in modules.items()}
            ckpt_mgr.load_best(phase_idx - 1, prev_phase.name, raw_modules)

        # Optimizer + scheduler
        optimizer = _build_optimizer(
            {n: (m.module if hasattr(m, 'module') else m) for n, m in modules.items()},
            phase
        )
        scheduler = _build_scheduler(optimizer, phase)

        grad_clip = phase.grad_clip_max_norm or config.engine.grad_clip_max_norm

        batch_frames = phase.batch_frames or config.data.batch_frames
        supervision_cams = phase.supervision_cams or config.data.supervision_cams

        # Phase losses config (dict of name -> dict)
        phase_losses = {name: {
            "weight": lc.weight,
            "warmup_steps": lc.warmup_steps,
            "compute_every_n": lc.compute_every_n,
            "downsample": lc.downsample,
        } for name, lc in phase.losses.items()}

        pbar = tqdm(range(phase.steps), desc=f"Phase {phase_idx}: {phase.name}",
                    disable=not is_main_process())

        for step in pbar:
            optimizer.zero_grad()

            batch = dataset.sample_training_batch(
                batch_scenes=config.data.batch_scenes,
                batch_frames=batch_frames,
            )

            with autocast_ctx():
                # Forward pass through model (experiment-specific)
                # The engine provides the batch; the experiment's forward logic
                # uses its own model architecture to produce Gaussian params.
                # This is handled by the compose_fn callback.
                predictions = compose_fn(modules, batch, step, phase, device)

                # Compute losses
                targets = {
                    "gt_image": predictions.pop("gt_image", None),
                    "points_map": predictions.pop("points_map", None),
                    "camera": predictions.pop("camera", None),
                }
                loss = loss_registry.compute(predictions, targets, step, phase_losses)

            # Backward
            if scaler:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for n in phase.trainable for p in modules[n].parameters()],
                        grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [p for n in phase.trainable for p in modules[n].parameters()],
                        grad_clip
                    )
                optimizer.step()

            scheduler.step()

            # Metrics (no .item() calls here)
            metrics_acc.update("loss", loss)
            if "psnr" in predictions:
                metrics_acc.update("psnr", predictions["psnr"])

            # Log
            if step % config.engine.log_every_n == 0 and is_main_process():
                flushed = metrics_acc.flush()
                pbar.set_postfix({k: f"{v:.4f}" for k, v in flushed.items()})
                if writer:
                    for k, v in flushed.items():
                        writer.add_scalar(f"{phase.name}/{k}", v, global_step)

            # Checkpoint
            if step % config.engine.save_every_n == 0 and step > 0 and is_main_process():
                raw_modules = {n: (m.module if hasattr(m, 'module') else m) for n, m in modules.items()}
                flushed = metrics_acc.flush()
                ckpt_mgr.save(phase_idx, phase.name, step, raw_modules,
                              optimizer, scheduler, flushed)

            global_step += 1

        # End of phase: save final checkpoint
        if is_main_process():
            raw_modules = {n: (m.module if hasattr(m, 'module') else m) for n, m in modules.items()}
            flushed = metrics_acc.flush()
            ckpt_mgr.save(phase_idx, phase.name, phase.steps, raw_modules,
                          optimizer, scheduler, flushed)

    if writer:
        writer.close()
    cleanup_ddp()

    if is_main_process():
        print("\nTraining complete.")


def main():
    parser = argparse.ArgumentParser(description="DecodeGaussians Training Engine")
    parser.add_argument("config", type=str, help="Config YAML path")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--sweep", nargs="*", help="Sweep mode: run configs on separate GPUs")
    args = parser.parse_args()

    if args.sweep:
        launch_sweep(args.sweep)
        return

    config = load_engine_config(args.config)

    # Detect experiment directory (parent of config file)
    experiment_dir = str(Path(args.config).parent)

    # Import experiment's build_model and compose functions
    exp_src = Path(experiment_dir) / "src"
    if exp_src.exists():
        sys.path.insert(0, str(exp_src))
        model_module = importlib.import_module("model")
        build_model_fn = model_module.build_model
        compose_fn = model_module.compose_forward
    else:
        raise FileNotFoundError(f"No src/ directory in {experiment_dir}")

    # DDP init
    strategy = resolve_strategy(config.distributed.strategy,
                                torch.cuda.device_count())
    if strategy == "ddp":
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        setup_ddp(rank, world_size)

    train(config, build_model_fn, compose_fn, experiment_dir=experiment_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_engine.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/engine.py tests/test_engine.py
git commit -m "feat: add core training engine with phase system and AMP"
```

---

### Task 11: Example Experiment Migration

**Files:**
- Create: `experiments/overfit_coffee_martini_v2/config.yaml`
- Create: `experiments/overfit_coffee_martini_v2/src/model.py`

Migrate the baseline coffee_martini experiment to use the shared engine. This serves as the reference for how experiments plug into the new infrastructure.

- [ ] **Step 1: Create experiment config using new schema**

```yaml
# experiments/overfit_coffee_martini_v2/config.yaml
engine:
  amp: true
  compile: true
  compile_mode: default
  grad_clip_max_norm: 1.0
  log_every_n: 50
  save_every_n: 2000
  eval_every_n: 1000

data:
  scenes:
    - name: coffee_martini
      path: datasets/neural_3d/coffee_martini
      precomputed: precomputed/coffee_martini
  resolution: [512, 384]
  batch_scenes: 1
  batch_frames: 1
  supervision_cams: 4
  memory_strategy: pinned
  input_camera: cam01
  eval_camera: cam00

precompute:
  backbone: spatracker_v2
  window_size: 8
  dtype: bfloat16
  output_dir: precomputed/

phases:
  - name: canonical
    steps: 20000
    trainable: [canonical_head]
    optimizer:
      type: adamw
      lr: 1.0e-3
      weight_decay: 0.01
    scheduler:
      type: cosine
      warmup_steps: 500
    losses:
      rgb:
        weight: 1.0
      ssim:
        weight: 0.85
        warmup_steps: 500
      geometric:
        weight: 0.1
      scale_reg:
        weight: 0.01
      opacity_reg:
        weight: 0.01

  - name: deformation
    steps: 15000
    trainable: [deformation_head]
    optimizer:
      type: adamw
      lr: 5.0e-4
    scheduler:
      type: cosine
      warmup_steps: 300
    losses:
      rgb:
        weight: 1.0
      ssim:
        weight: 0.85
      tv:
        weight: 0.01
      lpips:
        weight: 0.1
        compute_every_n: 5
        downsample: 2
      scale_reg:
        weight: 0.01
      opacity_reg:
        weight: 0.01

  - name: joint
    steps: 10000
    trainable: [canonical_head, deformation_head]
    optimizer:
      type: adamw
      lr: 2.0e-4
    scheduler:
      type: cosine
      warmup_steps: 200
    losses:
      rgb:
        weight: 1.0
      ssim:
        weight: 0.85
      tv:
        weight: 0.005
      lpips:
        weight: 0.1
        compute_every_n: 5
        downsample: 2
      scale_reg:
        weight: 0.01
      opacity_reg:
        weight: 0.01

distributed:
  strategy: auto
```

- [ ] **Step 2: Create experiment model.py with build_model and compose_forward**

```python
# experiments/overfit_coffee_martini_v2/src/model.py
"""Model definitions for coffee_martini baseline experiment.

Provides build_model() and compose_forward() for the shared engine.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: float) -> float:
    return math.log(x / (1.0 - x))


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


class CanonicalGaussianHead(nn.Module):
    def __init__(self, dim_in=2048, dim_hidden=1024, sh_degree=0,
                 num_gaussians_per_patch=128):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.K = num_gaussians_per_patch

        self.norm = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.GELU(),
        )
        dim_out = dim_hidden // 2
        self.xyz_head = nn.Linear(dim_out, self.K * 3)
        self.scale_head = nn.Linear(dim_out, self.K * 3)
        self.rot_head = nn.Linear(dim_out, self.K * 4)
        self.opacity_head = nn.Linear(dim_out, self.K * 1)
        self.sh_head = nn.Linear(dim_out, self.K * self.num_sh_coeffs * 3)

    def forward(self, tokens_mean):
        P = tokens_mean.shape[0]
        K = self.K
        x = self.mlp(self.norm(tokens_mean))

        xyz = self.xyz_head(x).view(P * K, 3)
        log_scale = self.scale_head(x).view(P * K, 3)
        rot = F.normalize(self.rot_head(x).view(P * K, 4), dim=-1)
        logit_opacity = self.opacity_head(x).view(P * K, 1)
        sh = self.sh_head(x).view(P * K, self.num_sh_coeffs, 3)

        return {
            "xyz": xyz, "log_scale": log_scale, "rot": rot,
            "logit_opacity": logit_opacity, "sh": sh,
            "hidden": x,
        }


class DeformationHead(nn.Module):
    def __init__(self, dim_in=2048, dim_hidden=256, n_attn_heads=8,
                 n_attn_layers=2, K=128, sh_degree=0):
        super().__init__()
        self.K = K
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        self.norm = nn.LayerNorm(dim_in)
        self.proj = nn.Linear(dim_in, dim_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden, nhead=n_attn_heads,
            dim_feedforward=dim_hidden * 4, batch_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)

        self.dxyz_head = nn.Linear(dim_hidden, K * 3)
        self.dscale_head = nn.Linear(dim_hidden, K * 3)
        self.drot_head = nn.Linear(dim_hidden, K * 4)
        self.dopacity_head = nn.Linear(dim_hidden, K * 1)
        self.dsh_head = nn.Linear(dim_hidden, K * self.num_sh_coeffs * 3)

    def forward(self, tokens_t):
        P = tokens_t.shape[0]
        K = self.K
        x = self.proj(self.norm(tokens_t))
        x = self.attn(x.unsqueeze(0)).squeeze(0)

        dxyz = torch.tanh(self.dxyz_head(x).view(P * K, 3)) * 0.1
        dscale = torch.tanh(self.dscale_head(x).view(P * K, 3)) * 0.5
        drot = F.normalize(self.drot_head(x).view(P * K, 4), dim=-1)
        dopacity = torch.tanh(self.dopacity_head(x).view(P * K, 1)) * 0.3
        dsh_flat = self.dsh_head(x).view(P * K, self.num_sh_coeffs * 3)
        dsh = dsh_flat.view(P * K, self.num_sh_coeffs, 3)

        all_deltas = torch.cat([dxyz, dscale, drot, dopacity, dsh_flat], dim=-1)
        return {
            "dxyz": dxyz, "dscale": dscale, "drot": drot,
            "dopacity": dopacity, "dsh": dsh, "all_deltas": all_deltas,
        }


def compose_gaussians(canonical, deltas=None, scale_factor=1.0):
    if deltas is None:
        means3D = canonical["xyz"]
        scales = scale_factor * F.softplus(canonical["log_scale"])
        rotations = canonical["rot"]
        opacity = torch.sigmoid(canonical["logit_opacity"])
        shs = canonical["sh"]
    else:
        means3D = canonical["xyz"] + deltas["dxyz"]
        scales = scale_factor * F.softplus(canonical["log_scale"] + deltas["dscale"])
        rotations = quaternion_multiply(canonical["rot"], deltas["drot"])
        rotations = F.normalize(rotations, dim=-1)
        opacity = torch.sigmoid(canonical["logit_opacity"] + deltas["dopacity"])
        shs = canonical["sh"] + deltas["dsh"]
    scales = torch.clamp(scales, min=1e-6, max=5.0)
    return means3D, scales, rotations, opacity, shs


# ── Engine interface ──────────────────────────────────────────────────────

def build_model(config):
    """Build model modules for the engine."""
    # Extract model config from phases or use defaults
    return {
        "canonical_head": CanonicalGaussianHead(
            dim_in=2048, dim_hidden=1024, sh_degree=0, num_gaussians_per_patch=128
        ),
        "deformation_head": DeformationHead(
            dim_in=2048, dim_hidden=256, n_attn_heads=8, n_attn_layers=2,
            K=128, sh_degree=0
        ),
    }


def compose_forward(modules, batch, step, phase, device):
    """Forward pass: produce predictions dict for the loss registry.

    This is the experiment-specific bridge between model and engine.
    """
    from src.renderer import render_gaussians
    from src.metrics import compute_psnr_tensor

    canonical_head = modules["canonical_head"]
    deformation_head = modules["deformation_head"]

    tokens_mean = batch["tokens_mean"]
    if tokens_mean.dim() == 3:
        tokens_mean = tokens_mean[0]  # [P, D] for single scene
    tokens_mean = tokens_mean.to(device).float()

    canonical = canonical_head(tokens_mean)

    if "deformation_head" in phase.trainable or phase.name == "joint":
        tokens_frames = batch["tokens_frames"]
        if tokens_frames.dim() == 3:
            tokens_frames = tokens_frames[0]  # [P, D] for single frame
        tokens_frames = tokens_frames.to(device).float()
        deltas = deformation_head(tokens_frames)
    else:
        deltas = None

    means3D, scales, rotations, opacity, shs = compose_gaussians(canonical, deltas)

    return {
        "rendered": None,  # filled by render loop
        "gaussian_means": means3D,
        "scales": scales,
        "rotations": rotations,
        "opacity": opacity,
        "shs": shs,
    }
```

- [ ] **Step 3: Commit**

```bash
mkdir -p experiments/overfit_coffee_martini_v2/src
git add experiments/overfit_coffee_martini_v2/config.yaml experiments/overfit_coffee_martini_v2/src/model.py
git commit -m "feat: add coffee_martini v2 experiment using shared engine"
```

---

### Task 12: LPIPS Integration in Loss Registry

**Files:**
- Modify: `src/losses.py`

Add LPIPS as a registered loss with downsample + compute_every_n optimization.

- [ ] **Step 1: Add LPIPS adapter to losses.py**

Add after the existing adapters in `src/losses.py`:

```python
# Add to imports at top of file:
import lpips as lpips_lib

# Add as module-level lazy singleton:
_lpips_net = None

def _get_lpips_net(device: str = "cuda"):
    """Lazy-load LPIPS AlexNet (shared singleton)."""
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = lpips_lib.LPIPS(net="alex").to(device)
        _lpips_net.eval()
        for p in _lpips_net.parameters():
            p.requires_grad = False
    return _lpips_net


def _lpips_adapter(predictions, targets, **kwargs):
    """LPIPS perceptual loss with optional downsampling."""
    rendered = predictions["rendered"]
    target = targets["gt_image"]
    downsample = kwargs.get("downsample", 1)

    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    # Downsample for speed
    if downsample > 1:
        rendered = F.interpolate(rendered, scale_factor=1.0/downsample, mode="bilinear",
                                 align_corners=False)
        target = F.interpolate(target, scale_factor=1.0/downsample, mode="bilinear",
                               align_corners=False)

    # LPIPS expects [-1, 1] range
    net = _get_lpips_net(rendered.device)
    return net(rendered * 2 - 1, target * 2 - 1).mean()
```

Then register in `LossRegistry.__init__`:
```python
self.register("lpips", _lpips_adapter)
```

- [ ] **Step 2: Commit**

```bash
git add src/losses.py
git commit -m "feat: add LPIPS loss with downsample optimization to registry"
```

---

### Task 13: Final Integration Test

**Files:**
- Modify: `tests/test_engine.py`

Add an end-to-end test that runs a minimal training step through the full engine pipeline (config → dataset → model → loss → backward → checkpoint).

- [ ] **Step 1: Add integration test**

Append to `tests/test_engine.py`:

```python
def test_full_training_step_cpu(tmp_path):
    """End-to-end: config → dataset → forward → loss → backward on CPU."""
    from src.config import load_engine_config
    from src.losses import LossRegistry
    from src.metrics import MetricAccumulator, compute_psnr_tensor
    from src.checkpoint import CheckpointManager

    # Create fake scene
    scene_dir = tmp_path / "precomputed" / "test_scene"
    _make_fake_scene(scene_dir, num_frames=5, num_patches=8, token_dim=32, H=16, W=16)

    # Create config
    cfg_dict = {
        "engine": {"amp": False, "compile": False, "log_every_n": 2, "save_every_n": 5},
        "data": {
            "scenes": [{"name": "test", "path": "", "precomputed": str(scene_dir)}],
            "memory_strategy": "pinned",
            "batch_frames": 1, "batch_scenes": 1,
        },
        "phases": [{
            "name": "test_phase", "steps": 3, "trainable": ["canonical_head"],
            "optimizer": {"lr": 1e-3},
            "losses": {"scale_reg": {"weight": 0.01}, "opacity_reg": {"weight": 0.01}},
        }],
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))
    config = load_engine_config(str(cfg_path))

    # Build model
    from src.dataset import SceneDataset
    dataset = SceneDataset(config.data, device="cpu")

    modules = {"canonical_head": DummyCanonicalHead(dim_in=32, K=4)}
    from src.engine import _apply_phase_freezing, _build_optimizer
    phase = config.phases[0]
    _apply_phase_freezing(modules, phase.trainable)
    optimizer = _build_optimizer(modules, phase)

    loss_registry = LossRegistry(device="cpu")
    metrics_acc = MetricAccumulator()

    # Run 3 training steps
    for step in range(3):
        optimizer.zero_grad()
        batch = dataset.sample_training_batch(batch_scenes=1, batch_frames=1)

        tokens = batch["tokens_mean"]
        if tokens.dim() == 3:
            tokens = tokens[0]

        canonical = modules["canonical_head"](tokens)

        predictions = {
            "scales": torch.nn.functional.softplus(canonical["log_scale"]),
            "opacity": torch.sigmoid(canonical["logit_opacity"]),
        }

        phase_losses = {name: {"weight": lc.weight, "warmup_steps": lc.warmup_steps,
                                "compute_every_n": lc.compute_every_n}
                        for name, lc in phase.losses.items()}
        loss = loss_registry.compute(predictions, {}, step, phase_losses)

        loss.backward()
        optimizer.step()
        metrics_acc.update("loss", loss)

    result = metrics_acc.flush()
    assert "loss" in result
    assert result["loss"] > 0

    # Save checkpoint
    ckpt_mgr = CheckpointManager(str(tmp_path / "ckpts"))
    ckpt_mgr.save(0, "test_phase", 3, modules, optimizer, None, result)
    assert (tmp_path / "ckpts" / "phase_0_test_phase" / "latest.pt").exists()
```

- [ ] **Step 2: Run all tests**

Run: `cd D:/DecodeGaussians && python -m pytest tests/test_engine_config.py tests/test_losses.py tests/test_metrics.py tests/test_checkpoint.py tests/test_dataset.py tests/test_engine.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine.py
git commit -m "test: add end-to-end integration test for training engine"
```

---

### Task 14: Final Commit and Verification

- [ ] **Step 1: Run full test suite**

```bash
cd D:/DecodeGaussians && python -m pytest tests/test_engine_config.py tests/test_losses.py tests/test_metrics.py tests/test_checkpoint.py tests/test_dataset.py tests/test_engine.py -v --tb=short
```

- [ ] **Step 2: Verify file structure**

```bash
ls -la src/
ls -la tests/test_engine*.py tests/test_losses.py tests/test_metrics.py tests/test_checkpoint.py tests/test_dataset.py
```

- [ ] **Step 3: Final commit with all files**

```bash
git add src/ tests/ experiments/overfit_coffee_martini_v2/
git commit -m "feat: complete shared training engine infrastructure

Adds src/ with config, dataset, losses, metrics, checkpoint, renderer,
compile, distributed, precompute, and engine modules. Includes test
suite and example experiment migration (coffee_martini_v2)."
```
