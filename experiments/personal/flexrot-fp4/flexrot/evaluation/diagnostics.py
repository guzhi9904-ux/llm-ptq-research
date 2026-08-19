from __future__ import annotations

import torch


def tensor_diagnostics(values: torch.Tensor) -> dict[str, float]:
    """返回结果 JSON 中常用的有限性、极值、均方根和零比例诊断。"""

    finite = torch.isfinite(values)
    return {
        "finite_fraction": float(finite.float().mean().item()),
        "amax": float(values.abs().max().item()),
        "rms": float(values.double().square().mean().sqrt().item()),
        "zero_fraction": float(values.eq(0).float().mean().item()),
    }
