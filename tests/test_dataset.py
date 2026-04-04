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
    ds = SceneDataset(config, device="cpu")
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
