# Shared Training Engine Design Spec

**Date:** 2026-04-04
**Goal:** Build a shared, GPU-optimized training engine in `src/` that serves both single-scene overfitting and multi-scene generalization training, with hardware-adaptive distributed support.

---

## 1. Directory Structure

```
D:/DecodeGaussians/src/
├── engine.py          # Core training loop, phase system, DDP orchestration
├── dataset.py         # Multi-scene dataset with configurable memory strategy
├── renderer.py        # Render wrapper (single + batched camera support)
├── losses.py          # Loss registry with LPIPS optimization
├── compile.py         # torch.compile wrappers for encoder/heads
├── distributed.py     # DDP setup, device detection, multi-GPU adaptation
├── config.py          # Config schema with phase definitions + memory strategy
├── metrics.py         # PSNR/SSIM/LPIPS eval, async-friendly (no per-step .item())
├── checkpoint.py      # Phase-aware save/resume, best-checkpoint tracking
└── precompute.py      # Shared STV2 precomputation pipeline
```

Experiments import from `src/` and provide their own `model.py` + `config.yaml`. Existing experiment directories remain untouched as historical artifacts.

---

## 2. Core Engine (`engine.py`)

### Training Loop

The engine runs a generalized phase-based training loop:

```
for phase in config.phases:
    freeze/unfreeze params per phase.trainable
    create optimizer + scheduler for phase
    for step in range(phase.steps):
        batch = dataset.sample_training_batch(...)
        with autocast (if amp enabled):
            predictions = model.forward(batch)
            loss = loss_registry.compute(predictions, batch, step, phase)
        scaler.scale(loss).backward()
        clip gradients
        scaler.step(optimizer)
        scaler.update()
        if step % log_every_n == 0: sync and log metrics
        if step % save_every_n == 0: save checkpoint
        if step % eval_every_n == 0: run eval
```

### Key Features

- **AMP by default**: `torch.amp.autocast("cuda", dtype=torch.float16)` + `GradScaler`. Togglable via `engine.amp: true/false`.
- **Logging decoupled from GPU**: losses accumulated on GPU tensors, synced to CPU every `log_every_n` steps (default 50). No `.item()` calls in the hot path.
- **Gradient clipping**: configurable `max_norm` per phase or global.
- **Gradient checkpointing**: opt-in per module via config flag. Applied to transformer/attention layers in deformation head.
- **Phase transitions**: automatic checkpoint save at phase boundaries, optimizer state reset, LR scheduler reset.

---

## 3. Generalized Phase System

Phases replace the current hardcoded 3-stage system. Each phase defines:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Phase identifier (for checkpoints, logging) |
| `steps` | int | Number of training steps |
| `trainable` | list[str] | Module names to unfreeze (all others frozen) |
| `optimizer` | dict | Optimizer type, LR, weight decay, betas |
| `scheduler` | dict | LR scheduler type, warmup steps, decay |
| `losses` | dict | Loss name → config (weight, warmup, compute_every_n, etc.) |
| `batch_frames` | int (optional) | Override global batch_frames for this phase |
| `supervision_cams` | int (optional) | Override camera count for this phase |

Phases execute sequentially. Each phase loads the best or latest checkpoint from the previous phase (configurable). The current 3-stage training is expressed as 3 phases in config — but experiments can define 1, 2, 5, or any number of phases.

---

## 4. Dataset (`dataset.py`)

### Memory Strategy

Four configurable modes for managing GPU memory, selectable via `data.memory_strategy`:

| Mode | Token Storage | GT Images | Best For |
|------|--------------|-----------|----------|
| `pinned` | float32 on GPU | All cached on GPU | Single-scene overfit (default) |
| `compressed` | float16 on GPU | All cached on GPU | 2-5 scenes, tight VRAM |
| `rotation` | N scenes pinned (LRU eviction) | Pinned for active set | 5-20 scenes |
| `streaming` | Async prefetch from CPU/disk | Prefetched per step | 100+ scenes |

- **`auto` mode** (default): selects strategy based on scene count + detected VRAM:
  - 1 scene → `pinned`
  - 2-5 scenes + enough VRAM → `compressed`
  - Otherwise → `streaming`
- **All modes configurable independently**: compression on/off (`data.token_dtype`), rotation budget (`data.rotation_budget_gb`), streaming prefetch workers
- **GT images always cached when possible**: eliminates per-step JPEG I/O. Falls back to disk loading only when VRAM budget is exhausted.
- **Multi-scene batching**: `data.batch_scenes` controls how many scenes per training step. Each scene contributes `batch_frames` frames × `supervision_cams` cameras.

### Interface

```python
class SceneDataset:
    """Unified dataset for single and multi-scene training."""

    def __init__(self, config, device):
        # Auto-detect memory strategy if auto
        # Load scene manifests, set up caching/streaming

    def sample_training_batch(self, batch_scenes, batch_frames, supervision_cams):
        """Returns dict with tokens, points_map, gt_images, cameras, scene_ids."""

    def get_eval_batch(self, scene_name, frame_idx, camera_name):
        """Returns single frame for evaluation."""

    def pin_scenes(self, scene_names):
        """Explicitly pin scenes to GPU (for rotation mode)."""

    def release_scenes(self, scene_names):
        """Release scenes from GPU (for rotation mode)."""
```

---

## 5. Precompute Pipeline (`precompute.py`)

Shared entry point for extracting and caching SpaTrackerV2 latents.

### Usage

```bash
# Single scene
python -m src.precompute --config experiments/my_exp/config.yaml

# Batch of scenes
python -m src.precompute --scenes datasets/neural_3d/* --output precomputed/

# On cluster: parallel across GPUs
python -m src.precompute --scenes datasets/neural_3d/* --output precomputed/ --parallel
```

### Config Section

```yaml
precompute:
  backbone: spatracker_v2
  window_size: 8
  dtype: bfloat16
  output_dir: precomputed/
  outputs:
    tokens: true        # [T, P, 2048] - STV2 aggregated tokens
    points_map: true    # [T, H, W, 3] - 3D point clouds
    poses: true         # [T, 4, 4] - camera poses
    intrinsics: true    # [T, 4, 4] - intrinsic matrices
    uncertainty: true   # [T, H, W] - uncertainty metric
```

### Features

- **Idempotent**: skips scenes with complete cache (checks `manifest.json`)
- **Manifest**: each scene's precomputed dir contains `manifest.json` with backbone version, resolution, frame count, output list, checksums, timestamp
- **OOM retry**: window-halving strategy (inherited from current implementation)
- **Extensible outputs**: the `outputs` dict maps to extractor functions. Adding a new cached tensor (e.g., DINOv2 features, depth maps from a different backbone) means registering one extractor function — no rewrite needed.
- **Multi-GPU batch**: on cluster, scenes distributed round-robin across available GPUs

### Output Structure

```
precomputed/
├── coffee_martini/
│   ├── manifest.json
│   ├── tokens.pt          # [300, 1041, 2048]
│   ├── points_map.pt      # [300, 384, 512, 3]
│   ├── poses.pt           # [300, 4, 4]
│   ├── intrinsics.pt      # [300, 4, 4]
│   └── uncertainty.pt     # [300, 384, 512]
├── flame_steak/
│   ├── manifest.json
│   └── ...
```

---

## 6. Loss Registry (`losses.py`)

### Architecture

```python
class LossRegistry:
    """Named loss functions with per-phase configuration."""

    def register(self, name: str, loss_fn: Callable): ...
    def compute(self, predictions, batch, step, phase_config) -> Tensor: ...
```

### Built-in Losses

| Name | Description | Configurable Options |
|------|-------------|---------------------|
| `rgb` | L1 photometric | `weight` |
| `ssim` | Structural similarity | `weight`, `warmup_steps` |
| `lpips` | AlexNet perceptual | `weight`, `warmup_steps`, `compute_every_n` (default 5), `downsample` (default 2) |
| `geometric` | Chamfer distance to STV2 point cloud | `weight`, `subsample_points` |
| `depth` | Rendered vs GT depth | `weight` |
| `tv` | Temporal variation on deformation deltas | `weight` |
| `scale_reg` | L1 on Gaussian scales | `weight` |
| `opacity_reg` | Entropy regularization | `weight` |

### Key Design Decisions

- **Opt-in per phase**: only losses listed in a phase's config are active. Omitted = disabled. Different scene types or experiments activate only what they need.
- **Per-loss config**: each loss has independent weight, warmup, compute frequency. No global coupling.
- **LPIPS optimization**: 2x spatial downsample before AlexNet + compute every N steps (default 5). Between computed steps, the last LPIPS value is held constant. Saves ~80% of LPIPS cost.
- **Custom loss registration**: experiments register additional losses via a `register_losses(registry)` hook in their `src/losses.py`.

---

## 7. Renderer (`renderer.py`)

Thin wrapper around `diff_gaussian_rasterization` CUDA kernel:

- **Single-camera render**: current interface, unchanged
- **Multi-camera render**: loops over cameras but shares Gaussian param tensors (no redundant copies)
- **Background color**: configurable (white/black/random for augmentation)
- **No torch.compile on rasterizer**: the CUDA extension is already optimized and compile can't trace custom kernels

---

## 8. Compile (`compile.py`)

```python
def compile_modules(model_dict: dict[str, nn.Module], config) -> dict[str, nn.Module]:
    """Apply torch.compile to encoder/head modules."""
    # Compiles: canonical_head, deformation_head (encoder + MLP)
    # Skips: anything that calls custom CUDA ops (rasterizer)
    # Mode: config.engine.compile_mode (default/reduce-overhead/max-autotune)
```

- Applied to canonical head MLP and deformation head (transformer encoder + MLPs)
- **Not applied** to the CUDA rasterizer
- Togglable via `engine.compile: true/false`
- Mode configurable: `"default"` for safe, `"max-autotune"` for best throughput after warmup

---

## 9. Distributed (`distributed.py`)

### Hardware Detection

Auto-detects at launch:
- GPU count, VRAM per device, CUDA compute capability
- Sets strategy accordingly (overridable in config)

### Strategies

| Strategy | When | Behavior |
|----------|------|----------|
| `single` | 1 GPU (4070S local) | No DDP overhead, direct training |
| `ddp` | Multi-GPU joint training | `DistributedDataParallel` with `nccl` backend, gradient sync across GPUs |
| `sweep` | Multi-GPU overfit experiments | Each GPU runs independent experiment, no gradient sync |
| `auto` | Default | `single` if 1 GPU, `ddp` if multi-GPU joint training |

### Future: Pipeline Parallelism

The module API is designed to support future pipeline parallelism:
- `build_model()` returns named modules → can be placed on different devices
- Phase system already tracks which modules are active → natural split points
- Not implemented now, but the abstractions don't prevent it

---

## 10. Checkpoint (`checkpoint.py`)

### Phase-Aware Checkpointing

- Saves at phase boundaries + every `save_every_n` steps
- Each checkpoint contains: model state, optimizer state, scheduler state, current phase index, step within phase, best metric
- **Phase transitions**: loads best checkpoint from previous phase (configurable: best vs. latest)
- **Resume**: `--resume` flag detects which phase/step to resume from

### Checkpoint Structure

```
checkpoints/
├── phase_0_canonical/
│   ├── step_20000.pt      # Final
│   ├── best.pt            # Best eval PSNR
│   └── latest.pt          # Symlink to most recent
├── phase_1_deformation/
│   ├── step_15000.pt
│   ├── best.pt
│   └── latest.pt
└── phase_2_joint/
    └── ...
```

---

## 11. Metrics (`metrics.py`)

- **Training metrics**: accumulated on GPU tensors, synced every `log_every_n` steps
- **Eval metrics**: PSNR, SSIM, LPIPS computed on eval camera, logged per eval step
- **No `.item()` in hot path**: eliminates GPU sync stalls during training
- **TensorBoard integration**: writes decoupled from training loop

---

## 12. Experiment Override Pattern

An experiment provides minimal overrides:

```
experiments/my_experiment/
├── config.yaml         # Extends/overrides base config
└── src/
    ├── model.py        # REQUIRED: build_model(config) -> dict[str, nn.Module]
    └── losses.py       # OPTIONAL: register_losses(registry) hook
```

### Model Contract

```python
def build_model(config) -> dict[str, nn.Module]:
    """Return named modules. Names must match phase.trainable references."""
    return {
        "canonical_head": CanonicalGaussianHead(config),
        "deformation_head": DeformationHead(config),
    }
```

### Training Launch

```bash
# Single-scene overfit
python -m src.engine experiments/my_experiment/config.yaml

# Multi-scene training
python -m src.engine configs/multi_scene.yaml

# Cluster DDP
torchrun --nproc_per_node=4 -m src.engine configs/multi_scene.yaml

# Cluster sweep (4 experiments in parallel)
python -m src.engine --sweep experiments/exp_a/config.yaml experiments/exp_b/config.yaml ...

# Resume
python -m src.engine experiments/my_experiment/config.yaml --resume
```

---

## 13. Example Configs

### Single-Scene Overfit (Current Behavior)

```yaml
engine:
  amp: true
  compile: true
  compile_mode: default
  grad_clip_max_norm: 1.0
  log_every_n: 50
  save_every_n: 2000
  eval_every_n: 1000
  gradient_checkpointing: false

data:
  scenes:
    - name: coffee_martini
      path: datasets/neural_3d/coffee_martini
      precomputed: precomputed/coffee_martini
  resolution: [512, 384]
  batch_scenes: 1
  batch_frames: 1
  supervision_cams: 4
  memory_strategy: auto
  token_dtype: float32

precompute:
  backbone: spatracker_v2
  window_size: 8
  dtype: bfloat16
  output_dir: precomputed/
  outputs:
    tokens: true
    points_map: true
    poses: true
    intrinsics: true
    uncertainty: true

phases:
  - name: canonical
    steps: 20000
    trainable: [canonical_head]
    optimizer: { type: adamw, lr: 1.0e-3, weight_decay: 0.01 }
    scheduler: { type: cosine, warmup_steps: 500 }
    losses:
      rgb: { weight: 1.0 }
      ssim: { weight: 0.85, warmup_steps: 500 }
      geometric: { weight: 0.1 }
      scale_reg: { weight: 0.01 }
      opacity_reg: { weight: 0.01 }

  - name: deformation
    steps: 15000
    trainable: [deformation_head]
    optimizer: { type: adamw, lr: 5.0e-4 }
    scheduler: { type: cosine, warmup_steps: 300 }
    losses:
      rgb: { weight: 1.0 }
      ssim: { weight: 0.85 }
      tv: { weight: 0.01 }
      lpips: { weight: 0.1, compute_every_n: 5, downsample: 2 }
      scale_reg: { weight: 0.01 }
      opacity_reg: { weight: 0.01 }

  - name: joint
    steps: 10000
    trainable: [canonical_head, deformation_head]
    optimizer: { type: adamw, lr: 2.0e-4 }
    scheduler: { type: cosine, warmup_steps: 200 }
    losses:
      rgb: { weight: 1.0 }
      ssim: { weight: 0.85 }
      tv: { weight: 0.005 }
      lpips: { weight: 0.1, compute_every_n: 5, downsample: 2 }
      scale_reg: { weight: 0.01 }
      opacity_reg: { weight: 0.01 }

distributed:
  strategy: auto
```

### Multi-Scene Generalization

```yaml
data:
  scenes:
    - name: coffee_martini
      path: datasets/neural_3d/coffee_martini
      precomputed: precomputed/coffee_martini
    - name: flame_steak
      path: datasets/neural_3d/flame_steak
      precomputed: precomputed/flame_steak
    # ... 100+ scenes
  batch_scenes: 4
  batch_frames: 2
  supervision_cams: 3
  memory_strategy: streaming
  token_dtype: float16

distributed:
  strategy: ddp  # 4x A5000
```

---

## 14. Acceleration Summary

| Optimization | Where | Expected Impact |
|-------------|-------|----------------|
| AMP (float16 autocast) | engine.py | 1.5-2x throughput |
| GT image caching | dataset.py | Eliminates ~1s/step I/O |
| LPIPS downsample + every-N | losses.py | ~80% LPIPS cost reduction |
| torch.compile on heads | compile.py | 1.2-2x on MLP/attention |
| Async logging (no .item()) | metrics.py | Eliminates GPU sync stalls |
| Gradient checkpointing | engine.py | Enables larger batches |
| Compressed tokens (fp16) | dataset.py | 2x more scenes in VRAM |
| DDP on cluster | distributed.py | Linear scaling across GPUs |

**Estimated single-scene overfit step time: ~200-500ms (down from 2-3s).**

---

## 15. What This Does NOT Cover (Future Work)

- Pipeline parallelism implementation (abstractions ready, not built)
- Automatic hyperparameter tuning
- Integration with `expmanager/` dashboard (will adapt later)
- New model architectures (experiments provide their own)
- Dataset download/preparation scripts
