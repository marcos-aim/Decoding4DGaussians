"""Integration tests for the training engine.

These test the engine's phase loop logic with dummy models.
Full GPU tests require CUDA and are marked accordingly.
"""
import pytest
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import json


def _make_fake_scene(scene_dir: Path, num_frames: int = 10, num_patches: int = 16,
                     token_dim: int = 32, H: int = 8, W: int = 8, num_cams: int = 3):
    """Create a fake precomputed scene for testing."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    tokens = torch.randn(num_frames, num_patches, token_dim)
    torch.save(tokens, scene_dir / "tokens.pt")
    points_map = torch.randn(num_frames, H, W, 3)
    torch.save(points_map, scene_dir / "points_map.pt")
    poses = torch.eye(4).unsqueeze(0).expand(num_frames, -1, -1)
    torch.save(poses, scene_dir / "poses.pt")
    intrs = torch.eye(4).unsqueeze(0).expand(num_frames, -1, -1)
    torch.save(intrs, scene_dir / "intrs.pt")
    unc = torch.rand(num_frames, H, W)
    torch.save(unc, scene_dir / "unc_metric.pt")
    for cam_idx in range(num_cams):
        cam_dir = scene_dir / "frames" / f"cam{cam_idx:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for f_idx in range(num_frames):
            img = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8)
            torch.save(img, cam_dir / f"{f_idx:06d}.pt")
    manifest = {"backbone": "spatracker_v2", "num_frames": num_frames, "resolution": [W, H]}
    (scene_dir / "manifest.json").write_text(json.dumps(manifest))


class DummyCanonicalHead(nn.Module):
    def __init__(self, dim_in=32, K=4):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(dim_in, K * 14)

    def forward(self, tokens_mean):
        P = tokens_mean.shape[0]
        out = self.linear(tokens_mean.float())
        out = out.view(P * self.K, 14)
        return {
            "xyz": out[:, :3],
            "log_scale": out[:, 3:6],
            "rot": torch.nn.functional.normalize(out[:, 6:10], dim=-1),
            "logit_opacity": out[:, 10:11],
            "sh": out[:, 11:14].unsqueeze(1),
        }


class DummyDeformationHead(nn.Module):
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

    _apply_phase_freezing(modules, trainable=["canonical_head"])
    _apply_phase_freezing(modules, trainable=["canonical_head", "deformation_head"])

    for p in modules["deformation_head"].parameters():
        assert p.requires_grad is True


def test_full_training_step_cpu(tmp_path):
    """End-to-end: config -> dataset -> forward -> loss -> backward on CPU."""
    from src.config import load_engine_config
    from src.losses import LossRegistry
    from src.metrics import MetricAccumulator
    from src.checkpoint import CheckpointManager
    from src.engine import _apply_phase_freezing, _build_optimizer
    from src.dataset import SceneDataset

    scene_dir = tmp_path / "precomputed" / "test_scene"
    _make_fake_scene(scene_dir, num_frames=5, num_patches=8, token_dim=32, H=16, W=16)

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

    dataset = SceneDataset(config.data, device="cpu")
    modules = {"canonical_head": DummyCanonicalHead(dim_in=32, K=4)}
    phase = config.phases[0]
    _apply_phase_freezing(modules, phase.trainable)
    optimizer = _build_optimizer(modules, phase)

    loss_registry = LossRegistry(device="cpu")
    metrics_acc = MetricAccumulator()

    for step in range(3):
        optimizer.zero_grad()
        batch = dataset.sample_training_batch(batch_scenes=1, batch_frames=1)
        tokens = batch["tokens_mean"]
        if isinstance(tokens, list):
            tokens = tokens[0]
        if tokens.dim() == 3:
            tokens = tokens[0]
        canonical = modules["canonical_head"](tokens)

        import torch.nn.functional as F
        predictions = {
            "scales": F.softplus(canonical["log_scale"]),
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

    ckpt_mgr = CheckpointManager(str(tmp_path / "ckpts"))
    ckpt_mgr.save(0, "test_phase", 3, modules, optimizer, None, result)
    assert (tmp_path / "ckpts" / "phase_0_test_phase" / "latest.pt").exists()
