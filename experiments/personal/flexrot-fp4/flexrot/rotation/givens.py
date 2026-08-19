from __future__ import annotations

import math

import torch


def givens_kernel(
    theta_deg: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """构造 FlexRot 使用的二维正交核 ``F(theta)``。

    这里的 ``theta`` 是一次蝶形混合的强度，而不是给整个隐藏向量指定
    一个普通二维平面旋转。核定义为 ``[[cosθ, sinθ], [sinθ, -cosθ]]``：
    两行单位范数且内积为零，因此对任意角度都保持正交。θ=45° 时两项
    等幅，成为归一化 H2；较小角度则减少跨通道能量混合。
    """

    theta = float(theta_deg)
    if not 0.0 <= theta <= 45.0:
        raise ValueError("theta_deg 必须位于 [0, 45]")
    radians = math.radians(theta)
    cosine, sine = math.cos(radians), math.sin(radians)
    return torch.tensor(
        [[cosine, sine], [sine, -cosine]], dtype=dtype, device=device
    )
