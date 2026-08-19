from __future__ import annotations

import math

import torch

from .givens import givens_kernel


def _validate_block_size(block_size: int) -> None:
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size 必须是正的 2 的幂")


def normalized_hadamard(
    block_size: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """按 Sylvester 递推构造自然顺序的归一化 Hadamard 矩阵。

    H16/H32 分别由 H2 做 4/5 次 tensor product（Kronecker 积）得到。
    每次积都保持 ``H.T @ H = I``，因此最终矩阵不会改变向量二范数。
    """

    _validate_block_size(block_size)
    # 数学上等价于归一化 H2 的重复 tensor product。实现上先递推 ±1
    # Sylvester 矩阵、最后只归一化一次，以逐位复现已验证 FP32 native 端点；
    # 若每层都乘 1/sqrt(2)，约 3e-8 的累计差异也可能改变 E2M1 舍入边界。
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < block_size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(float(block_size))


def parameterized_hadamard(
    block_size: int,
    theta_deg: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """构造 ``F(theta)^tensor(log2(block_size))`` 的可调正交矩阵。

    θ=45° 时 ``F`` 就是 H2，所以结果严格退化为 native Hadamard；
    θ<45° 时仍正交，但通道混合更弱。NVFP4 默认使用 H16 支持，MXFP4
    默认使用 H32 支持，使旋转块边界与各自量化块边界一致。
    """

    _validate_block_size(block_size)
    if float(theta_deg) == 45.0:
        return normalized_hadamard(
            block_size, dtype=dtype, device=device
        )
    kernel = givens_kernel(theta_deg, dtype=dtype, device=device)
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    for _ in range(block_size.bit_length() - 1):
        matrix = torch.kron(matrix, kernel)
    return matrix
