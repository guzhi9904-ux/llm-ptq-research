from __future__ import annotations

import torch

from .hadamard import parameterized_hadamard


def build_rotation(
    block_size: int,
    theta_deg: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """构造可调强度的正交旋转矩阵。

    ``theta=45°`` 时等价于标准 Hadamard mixing；更小角度表示更弱的
    通道混合。对 NVFP4 使用 block_size=16，对 MXFP4 使用 32。
    """

    return parameterized_hadamard(
        block_size, theta_deg, dtype=dtype, device=device
    )


@torch.no_grad()
def apply_rotation(
    values: torch.Tensor,
    *,
    block_size: int,
    theta_deg: float,
) -> torch.Tensor:
    """沿最后一维逐块右乘 FlexRot 矩阵，不改变输入张量形状。"""

    if values.ndim == 0 or values.shape[-1] % block_size:
        raise ValueError("最后一维必须能被 block_size 整除")
    rotation = build_rotation(
        block_size,
        theta_deg,
        dtype=values.dtype,
        device=values.device,
    )
    grouped = values.reshape(*values.shape[:-1], -1, block_size)
    return grouped.matmul(rotation).reshape_as(values)


@torch.no_grad()
def rotate_linear_problem(
    activations: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_size: int,
    theta_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """成对变换线性层输入 X 与权重 W，并保持未量化输出等价。

    对 ``Y=X W^T``，两边都右乘同一个正交矩阵 R：
    ``(X R)(W R)^T = X R R^T W^T = X W^T``。因此 FlexRot 只改变
    量化器看到的通道坐标，不改变 BF16/FP32 线性算子的数学函数。
    """

    if activations.shape[-1] != weight.shape[-1]:
        raise ValueError("activation 与 weight 的输入维度不一致")
    return (
        apply_rotation(
            activations, block_size=block_size, theta_deg=theta_deg
        ),
        apply_rotation(weight, block_size=block_size, theta_deg=theta_deg),
    )
