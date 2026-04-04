"""Config schema for shared training engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import yaml


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
    name: str = ""
    steps: int = 0
    trainable: list[str] = field(default_factory=list)
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
    sweep_configs: Optional[list[dict]] = None


@dataclass
class TrainingConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    precompute: PrecomputeConfig = field(default_factory=PrecomputeConfig)
    phases: list[PhaseConfig] = field(default_factory=list)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


def _parse_loss_config(d: dict) -> LossConfig:
    return LossConfig(
        weight=d.get("weight", 1.0),
        warmup_steps=d.get("warmup_steps", 0),
        compute_every_n=d.get("compute_every_n", 1),
        downsample=d.get("downsample", 1),
    )


def _parse_phase(d: dict) -> PhaseConfig:
    opt_dict = d.get("optimizer", {})
    optimizer = OptimizerConfig(
        type=opt_dict.get("type", "adamw"),
        lr=opt_dict.get("lr", 1e-3),
        weight_decay=opt_dict.get("weight_decay", 0.01),
        betas=tuple(opt_dict.get("betas", (0.9, 0.999))),
    )

    sched_dict = d.get("scheduler", {})
    scheduler = SchedulerConfig(
        type=sched_dict.get("type", "cosine"),
        warmup_steps=sched_dict.get("warmup_steps", 500),
    )

    losses = {
        k: _parse_loss_config(v) for k, v in d.get("losses", {}).items()
    }

    return PhaseConfig(
        name=d.get("name", ""),
        steps=d.get("steps", 0),
        trainable=d.get("trainable", []),
        optimizer=optimizer,
        scheduler=scheduler,
        losses=losses,
        batch_frames=d.get("batch_frames", None),
        supervision_cams=d.get("supervision_cams", None),
        grad_clip_max_norm=d.get("grad_clip_max_norm", None),
    )


def load_engine_config(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Engine config
    eng_dict = raw.get("engine", {})
    engine = EngineConfig(
        amp=eng_dict.get("amp", True),
        compile=eng_dict.get("compile", True),
        compile_mode=eng_dict.get("compile_mode", "default"),
        grad_clip_max_norm=eng_dict.get("grad_clip_max_norm", 1.0),
        log_every_n=eng_dict.get("log_every_n", 50),
        save_every_n=eng_dict.get("save_every_n", 2000),
        eval_every_n=eng_dict.get("eval_every_n", 1000),
        gradient_checkpointing=eng_dict.get("gradient_checkpointing", False),
    )

    # Data config
    data_dict = raw.get("data", {})
    scenes = [
        SceneConfig(
            name=s.get("name", ""),
            path=s.get("path", ""),
            precomputed=s.get("precomputed", ""),
        )
        for s in data_dict.get("scenes", [])
    ]
    res_raw = data_dict.get("resolution", (512, 384))
    resolution = tuple(res_raw) if not isinstance(res_raw, tuple) else res_raw
    data = DataConfig(
        scenes=scenes,
        resolution=resolution,
        batch_scenes=data_dict.get("batch_scenes", 1),
        batch_frames=data_dict.get("batch_frames", 1),
        supervision_cams=data_dict.get("supervision_cams", 4),
        memory_strategy=data_dict.get("memory_strategy", "auto"),
        token_dtype=data_dict.get("token_dtype", "float32"),
        rotation_budget_gb=data_dict.get("rotation_budget_gb", None),
        input_camera=data_dict.get("input_camera", "cam01"),
        eval_camera=data_dict.get("eval_camera", "cam00"),
    )

    # Precompute config
    pc_dict = raw.get("precompute", {})
    outputs_dict = pc_dict.get("outputs", {})
    outputs = PrecomputeOutputsConfig(
        tokens=outputs_dict.get("tokens", True),
        points_map=outputs_dict.get("points_map", True),
        poses=outputs_dict.get("poses", True),
        intrinsics=outputs_dict.get("intrinsics", True),
        uncertainty=outputs_dict.get("uncertainty", True),
    )
    precompute = PrecomputeConfig(
        backbone=pc_dict.get("backbone", "spatracker_v2"),
        window_size=pc_dict.get("window_size", 8),
        dtype=pc_dict.get("dtype", "bfloat16"),
        output_dir=pc_dict.get("output_dir", "precomputed/"),
        outputs=outputs,
    )

    # Phases
    phases = [_parse_phase(p) for p in raw.get("phases", [])]

    # Distributed config
    dist_dict = raw.get("distributed", {})
    distributed = DistributedConfig(
        strategy=dist_dict.get("strategy", "auto"),
        sweep_configs=dist_dict.get("sweep_configs", None),
    )

    return TrainingConfig(
        engine=engine,
        data=data,
        precompute=precompute,
        phases=phases,
        distributed=distributed,
    )
