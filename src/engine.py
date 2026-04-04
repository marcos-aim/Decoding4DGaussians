"""Core training engine with phase system, AMP, compile, DDP, and checkpointing.

Ties together: config, dataset, losses, metrics, checkpoint, compile_utils, distributed.
The renderer is imported lazily so CPU-only tests can run without diff_gaussian_rasterization.
"""
from __future__ import annotations

import argparse
import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR

from src.config import TrainingConfig, PhaseConfig, load_engine_config
from src.dataset import SceneDataset
from src.losses import LossRegistry
from src.metrics import MetricAccumulator
from src.checkpoint import CheckpointManager
from src.compile_utils import compile_modules
from src.distributed import (
    is_main_process,
    get_rank,
    get_world_size,
    setup_ddp,
    cleanup_ddp,
    resolve_strategy,
    detect_gpus,
    launch_sweep,
)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _apply_phase_freezing(
    modules: dict[str, nn.Module],
    trainable: list[str],
) -> None:
    """Freeze all modules, then unfreeze those listed in ``trainable``.

    Args:
        modules:   dict of name -> nn.Module
        trainable: list of module names that should remain trainable
    """
    trainable_set = set(trainable)

    for name, mod in modules.items():
        requires_grad = name in trainable_set
        for p in mod.parameters():
            p.requires_grad = requires_grad


def _build_optimizer(
    modules: dict[str, nn.Module],
    phase: PhaseConfig,
) -> torch.optim.Optimizer:
    """Build optimizer for trainable parameters only.

    Only parameters with ``requires_grad=True`` are included.

    Args:
        modules: dict of name -> nn.Module (some frozen, some not)
        phase:   PhaseConfig with optimizer sub-config

    Returns:
        Configured optimizer instance.
    """
    trainable_params = [
        p for mod in modules.values()
        for p in mod.parameters()
        if p.requires_grad
    ]

    opt_cfg = phase.optimizer
    opt_type = opt_cfg.type.lower()

    if opt_type == "adamw":
        return AdamW(
            trainable_params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=opt_cfg.betas,
        )
    elif opt_type == "adam":
        return Adam(
            trainable_params,
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
        )
    elif opt_type == "sgd":
        return SGD(
            trainable_params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type!r}. Choose from adamw, adam, sgd.")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    phase: PhaseConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build learning rate scheduler for the given phase.

    Supported types:
        - ``cosine``: CosineAnnealingLR over phase.steps
        - ``constant``: ConstantLR (no-op, factor=1.0)

    Args:
        optimizer: the optimizer to schedule
        phase:     PhaseConfig with scheduler sub-config

    Returns:
        Scheduler instance.
    """
    sched_cfg = phase.scheduler
    sched_type = sched_cfg.type.lower()

    if sched_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, phase.steps))
    elif sched_type == "constant":
        return ConstantLR(optimizer, factor=1.0, total_iters=phase.steps)
    else:
        # Fallback: constant LR
        return ConstantLR(optimizer, factor=1.0, total_iters=phase.steps)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    config: TrainingConfig,
    build_model_fn: Callable[[], dict[str, nn.Module]],
    compose_fn: Optional[Callable] = None,
    cameras: Optional[list] = None,
    experiment_dir: str = ".",
) -> None:
    """Main training loop.

    Args:
        config:         Parsed TrainingConfig
        build_model_fn: Callable() -> dict[str, nn.Module]
        compose_fn:     Optional callable to compose Gaussians for rendering
        cameras:        Optional list of camera objects for rendering
        experiment_dir: Root directory for logs and checkpoints
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = config.engine.amp and device == "cuda"
    use_compile = config.engine.compile and device == "cuda"

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    strategy = resolve_strategy(config.distributed.strategy, num_gpus)
    rank = get_rank()
    world_size = get_world_size()

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    modules = build_model_fn()
    for mod in modules.values():
        mod.to(device)

    # Optionally compile
    if use_compile:
        modules = compile_modules(
            modules,
            compile_enabled=True,
            compile_mode=config.engine.compile_mode,
            skip_names={"renderer"},
        )

    # Optionally wrap in DDP
    ddp_modules: dict[str, nn.Module] = {}
    if strategy == "ddp" and torch.cuda.is_available():
        from torch.nn.parallel import DistributedDataParallel as DDP
        for name, mod in modules.items():
            ddp_modules[name] = DDP(mod, device_ids=[rank])
    else:
        ddp_modules = modules

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------
    dataset = SceneDataset(config.data, device=device, cameras=cameras)
    loss_registry = LossRegistry(device=device)

    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    ckpt_mgr = CheckpointManager(ckpt_dir)

    log_dir = os.path.join(experiment_dir, "logs")
    writer = None
    if is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("[engine] TensorBoard not available; skipping SummaryWriter.")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ------------------------------------------------------------------
    # Resume: find the latest checkpoint
    # ------------------------------------------------------------------
    resume_phase_idx = 0
    resume_step = 0
    resume_info = ckpt_mgr.find_resume_point()
    if resume_info is not None:
        resume_phase_idx = resume_info["phase_idx"]
        resume_step = resume_info["step"]
        print(f"[engine] Resuming from phase {resume_phase_idx}, step {resume_step}")

    # ------------------------------------------------------------------
    # Phase loop
    # ------------------------------------------------------------------
    global_step = 0

    for phase_idx, phase in enumerate(config.phases):
        # Skip phases we've already completed
        if phase_idx < resume_phase_idx:
            global_step += phase.steps
            continue

        print(f"\n[engine] === Phase {phase_idx}: {phase.name} ({phase.steps} steps) ===")

        # Freeze/unfreeze
        _apply_phase_freezing(ddp_modules, phase.trainable)

        # Load best checkpoint from previous phase (warm-start weights)
        if phase_idx > 0:
            prev_phase = config.phases[phase_idx - 1]
            loaded = ckpt_mgr.load_best(phase_idx - 1, prev_phase.name, modules)
            if loaded is not None:
                print(f"[engine] Loaded best checkpoint from phase {phase_idx - 1}: {prev_phase.name}")

        # Resume within current phase if applicable
        start_step = 0
        if phase_idx == resume_phase_idx and resume_step > 0:
            start_step = resume_step
            ckpt_mgr.load_latest(phase_idx, phase.name, modules)
            print(f"[engine] Resuming phase {phase_idx} from step {start_step}")

        optimizer = _build_optimizer(ddp_modules, phase)
        scheduler = _build_scheduler(optimizer, phase)

        metrics_acc = MetricAccumulator()

        # Per-phase loss config dict
        phase_losses = {
            name: {
                "weight": lc.weight,
                "warmup_steps": lc.warmup_steps,
                "compute_every_n": lc.compute_every_n,
            }
            for name, lc in phase.losses.items()
        }

        grad_clip = (
            phase.grad_clip_max_norm
            if phase.grad_clip_max_norm is not None
            else config.engine.grad_clip_max_norm
        )

        batch_frames = phase.batch_frames or config.data.batch_frames
        batch_scenes = config.data.batch_scenes
        supervision_cams = phase.supervision_cams or config.data.supervision_cams

        # ------------------------------------------------------------------
        # Step loop
        # ------------------------------------------------------------------
        for step in range(start_step, phase.steps):
            optimizer.zero_grad(set_to_none=True)

            batch = dataset.sample_training_batch(
                batch_scenes=batch_scenes,
                batch_frames=batch_frames,
                supervision_cams=supervision_cams,
            )

            # Forward pass (with optional AMP)
            autocast_ctx = (
                torch.cuda.amp.autocast()
                if use_amp
                else torch.no_grad.__class__()  # dummy context manager that does nothing
            )

            # Use a simple context manager approach
            if use_amp:
                with torch.cuda.amp.autocast():
                    predictions, targets = _forward_pass(
                        batch, ddp_modules, phase, compose_fn, cameras, device
                    )
                    loss = loss_registry.compute(predictions, targets, step, phase_losses)
            else:
                predictions, targets = _forward_pass(
                    batch, ddp_modules, phase, compose_fn, cameras, device
                )
                loss = loss_registry.compute(predictions, targets, step, phase_losses)

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    for mod in ddp_modules.values():
                        nn.utils.clip_grad_norm_(mod.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    for mod in ddp_modules.values():
                        nn.utils.clip_grad_norm_(mod.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()

            # Accumulate metrics
            metrics_acc.update("loss", loss)

            # Logging
            log_every = config.engine.log_every_n
            if is_main_process() and step % log_every == 0:
                metrics = metrics_acc.flush()
                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(f"{phase.name}/{k}", v, global_step + step)
                lr = optimizer.param_groups[0]["lr"]
                loss_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"  [step {step:6d}] {loss_str}  lr={lr:.2e}")

            # Checkpointing
            save_every = config.engine.save_every_n
            if is_main_process() and step > 0 and step % save_every == 0:
                metrics_snapshot = metrics_acc.flush() if metrics_acc._sums else {}
                ckpt_mgr.save(
                    phase_idx, phase.name, step,
                    modules, optimizer, scheduler, metrics_snapshot
                )

        # End-of-phase checkpoint
        if is_main_process():
            final_metrics = metrics_acc.flush() if metrics_acc._sums else {}
            ckpt_mgr.save(
                phase_idx, phase.name, phase.steps,
                modules, optimizer, scheduler, final_metrics
            )
            print(f"[engine] Phase {phase_idx} ({phase.name}) complete. Checkpoint saved.")

        global_step += phase.steps

    # Cleanup
    if writer is not None:
        writer.close()
    cleanup_ddp()
    print("[engine] Training complete.")


# ---------------------------------------------------------------------------
# Forward pass helper (no rendering for phases without compose_fn/cameras)
# ---------------------------------------------------------------------------

def _forward_pass(
    batch: dict,
    modules: dict[str, nn.Module],
    phase: PhaseConfig,
    compose_fn: Optional[Callable],
    cameras: Optional[list],
    device: str,
) -> tuple[dict, dict]:
    """Run model forward pass for one batch.

    For phases without a compose_fn (e.g. canonical-only), just runs the
    canonical head and returns predictions dict.  For phases with compose_fn
    and cameras, also runs the deformation head and renderer.

    Returns:
        predictions: dict of model outputs
        targets:     dict of ground-truth data
    """
    predictions: dict = {}
    targets: dict = {}

    # tokens_mean is a list (one per scene in batch)
    tokens_mean_list = batch["tokens_mean"]
    tokens_frames_list = batch["tokens_frames"]

    # Use the first scene in the batch for simplicity
    tokens_mean = tokens_mean_list[0] if isinstance(tokens_mean_list, list) else tokens_mean_list
    tokens_frames = tokens_frames_list[0] if isinstance(tokens_frames_list, list) else tokens_frames_list

    if tokens_mean.dim() == 3:
        tokens_mean = tokens_mean[0]

    # Canonical head
    if "canonical_head" in modules:
        canonical = modules["canonical_head"](tokens_mean)
        import torch.nn.functional as F
        predictions["scales"] = F.softplus(canonical["log_scale"])
        predictions["opacity"] = torch.sigmoid(canonical["logit_opacity"])
        predictions["gaussian_means"] = canonical["xyz"]
        predictions["canonical"] = canonical

    # Deformation head + rendering (only when compose_fn and cameras are provided)
    if "deformation_head" in modules and compose_fn is not None and cameras is not None:
        # tokens_frames: [F, P, D] — use first frame
        tokens_t = tokens_frames[0] if tokens_frames.dim() == 3 else tokens_frames
        deformation = modules["deformation_head"](tokens_t)
        predictions["deformation"] = deformation
        predictions["all_deltas"] = deformation.get("all_deltas", tokens_t)

        # Compose and render lazily (renderer import here to avoid import errors in CPU tests)
        try:
            from src.renderer import render_gaussians
            canonical = predictions.get("canonical", {})
            composed = compose_fn(canonical, deformation)

            camera = cameras[0] if cameras else None
            if camera is not None:
                bg_color = torch.zeros(3, device=device)
                rendered, radii, depth, _ = render_gaussians(
                    means3D=composed["xyz"],
                    scales=composed["scales"],
                    rotations=composed["rot"],
                    opacity=composed["opacity"],
                    shs=composed["sh"],
                    camera=camera,
                    bg_color=bg_color,
                )
                predictions["rendered"] = rendered
                predictions["rendered_depth"] = depth
        except ImportError:
            pass  # Renderer not available; skip rendering

    return predictions, targets


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: train from a config file."""
    parser = argparse.ArgumentParser(description="DecodeGaussians training engine")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint in experiment_dir"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Launch a sweep across GPUs using distributed.sweep_configs"
    )
    parser.add_argument(
        "--experiment_dir", type=str, default=".",
        help="Root directory for logs and checkpoints"
    )
    args = parser.parse_args()

    config = load_engine_config(args.config)

    if args.sweep:
        sweep_cfgs = config.distributed.sweep_configs
        if not sweep_cfgs:
            raise ValueError("--sweep requires distributed.sweep_configs in config")
        import tempfile, yaml as _yaml
        config_paths = []
        tmpdir = tempfile.mkdtemp()
        for i, overrides in enumerate(sweep_cfgs):
            import copy
            sweep_config = copy.deepcopy(config)
            # For now just launch the same config on each GPU
            cfg_path = os.path.join(tmpdir, f"sweep_{i}.yaml")
            config_paths.append(args.config)
        launch_sweep(config_paths)
        return

    # Build model is a no-op here; caller should pass build_model_fn directly.
    # For CLI invocation, define a minimal stub that raises if called without model.
    def _no_model():
        raise RuntimeError(
            "main() CLI stub: supply build_model_fn to train() directly, "
            "or subclass the engine for your specific model."
        )

    train(
        config=config,
        build_model_fn=_no_model,
        experiment_dir=args.experiment_dir,
    )


if __name__ == "__main__":
    main()
