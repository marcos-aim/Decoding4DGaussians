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
    mse = torch.clamp(mse, min=1e-10)
    return -10.0 * torch.log10(mse)
