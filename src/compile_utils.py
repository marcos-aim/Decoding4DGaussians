# src/compile_utils.py
"""torch.compile wrappers for encoder/head modules.

Applied to: canonical_head, deformation_head (MLP + transformer encoder)
NOT applied to: diff_gaussian_rasterization (custom CUDA kernel)

Modes:
  - "default": safe, minimal overhead
  - "reduce-overhead": uses CUDA graphs, faster steady-state
  - "max-autotune": benchmarks kernels, best throughput after warmup
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compile_modules(
    modules: dict[str, nn.Module],
    compile_enabled: bool = True,
    compile_mode: str = "default",
    skip_names: set[str] | None = None,
) -> dict[str, nn.Module]:
    """Apply torch.compile to named modules.

    Args:
        modules: dict of name -> nn.Module
        compile_enabled: if False, return modules unchanged
        compile_mode: torch.compile mode
        skip_names: module names to skip (e.g., modules that call custom CUDA ops)

    Returns:
        Dict with same keys, compiled modules where applicable.
    """
    if not compile_enabled:
        return modules

    skip = skip_names or set()
    compiled = {}

    for name, module in modules.items():
        if name in skip:
            compiled[name] = module
            print(f"  [compile] Skipping {name} (in skip list)")
        else:
            try:
                compiled[name] = torch.compile(module, mode=compile_mode)
                print(f"  [compile] Compiled {name} (mode={compile_mode})")
            except Exception as e:
                print(f"  [compile] Failed to compile {name}: {e}, using eager mode")
                compiled[name] = module

    return compiled
