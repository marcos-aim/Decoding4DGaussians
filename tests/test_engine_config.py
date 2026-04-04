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
    assert cfg.engine.amp is True
    assert cfg.engine.compile is True
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
    assert phase.batch_frames is None
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
