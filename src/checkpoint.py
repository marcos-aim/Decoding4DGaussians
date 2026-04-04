"""Phase-aware checkpoint save/resume with best-metric tracking."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages phase-aware checkpoints with best-metric tracking.

    Checkpoints are organized as:
        checkpoint_dir/
            phase_{idx}_{name}/
                step_{N:06d}.pt
                latest.pt   (copy of most recent step checkpoint)
                best.pt     (copy of step checkpoint with best metric)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        best_metric: str = "psnr",
        higher_is_better: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_metric = best_metric
        self.higher_is_better = higher_is_better
        # Track best metric value per phase key
        self._best_values: dict[str, float] = {}

    def _phase_dir(self, phase_idx: int, phase_name: str) -> Path:
        """Return path to phase directory, creating it if needed."""
        phase_dir = self.checkpoint_dir / f"phase_{phase_idx}_{phase_name}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir

    def save(
        self,
        phase_idx: int,
        phase_name: str,
        step: int,
        modules: dict[str, nn.Module],
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        metrics: dict[str, float],
    ) -> Path:
        """Save a numbered checkpoint, update latest.pt, and update best.pt if improved."""
        phase_dir = self._phase_dir(phase_idx, phase_name)

        state = {
            "phase_idx": phase_idx,
            "phase_name": phase_name,
            "step": step,
            "metrics": metrics,
            "modules": {name: mod.state_dict() for name, mod in modules.items()},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }

        step_path = phase_dir / f"step_{step:06d}.pt"
        torch.save(state, step_path)

        # Update latest.pt
        latest_path = phase_dir / "latest.pt"
        shutil.copy2(step_path, latest_path)

        # Update best.pt if metric improved
        phase_key = f"{phase_idx}_{phase_name}"
        metric_value = metrics.get(self.best_metric)
        if metric_value is not None:
            best_value = self._best_values.get(phase_key)
            is_better = best_value is None or (
                metric_value > best_value if self.higher_is_better
                else metric_value < best_value
            )
            if is_better:
                self._best_values[phase_key] = metric_value
                best_path = phase_dir / "best.pt"
                shutil.copy2(step_path, best_path)

        return step_path

    def load_latest(
        self,
        phase_idx: int,
        phase_name: str,
        modules: dict[str, nn.Module],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict:
        """Load latest.pt for the given phase into modules/optimizer/scheduler."""
        phase_dir = self._phase_dir(phase_idx, phase_name)
        latest_path = phase_dir / "latest.pt"
        return self._load(latest_path, modules, optimizer, scheduler)

    def load_best(
        self,
        phase_idx: int,
        phase_name: str,
        modules: dict[str, nn.Module],
    ) -> dict | None:
        """Load best.pt for the given phase, or return None if it doesn't exist."""
        phase_dir = self._phase_dir(phase_idx, phase_name)
        best_path = phase_dir / "best.pt"
        if not best_path.exists():
            return None
        return self._load(best_path, modules)

    def find_resume_point(self) -> dict | None:
        """Scan all phase dirs and return the latest checkpoint info, or None."""
        if not self.checkpoint_dir.exists():
            return None

        phase_pattern = re.compile(r"phase_(\d+)_(.*)")
        best_phase_idx = -1
        best_step = -1
        best_phase_name = ""

        for phase_dir in self.checkpoint_dir.iterdir():
            if not phase_dir.is_dir():
                continue
            m = phase_pattern.match(phase_dir.name)
            if not m:
                continue
            phase_idx = int(m.group(1))
            phase_name = m.group(2)

            latest_path = phase_dir / "latest.pt"
            if not latest_path.exists():
                continue

            state = torch.load(latest_path, weights_only=False)
            step = state["step"]

            # Pick the phase with the highest idx; tie-break on step
            if phase_idx > best_phase_idx or (
                phase_idx == best_phase_idx and step > best_step
            ):
                best_phase_idx = phase_idx
                best_step = step
                best_phase_name = phase_name

        if best_phase_idx < 0:
            return None

        return {
            "phase_idx": best_phase_idx,
            "phase_name": best_phase_name,
            "step": best_step,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(
        self,
        path: Path,
        modules: dict[str, nn.Module],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> dict:
        state = torch.load(path, weights_only=False)

        for name, mod in modules.items():
            if name in state["modules"]:
                mod.load_state_dict(state["modules"][name])

        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])

        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])

        return {
            "phase_idx": state["phase_idx"],
            "phase_name": state["phase_name"],
            "step": state["step"],
            "metrics": state["metrics"],
        }
