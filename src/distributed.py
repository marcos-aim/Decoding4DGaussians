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
