from __future__ import annotations

import torch


def relative_sse(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """计算相对平方误差，分母为零时用极小正数保护。"""

    numerator = (actual.double() - reference.double()).square().sum()
    denominator = reference.double().square().sum().clamp_min(1e-30)
    return float((numerator / denominator).item())


def cosine_similarity(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = actual.double().flatten(), reference.double().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
