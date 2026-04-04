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
