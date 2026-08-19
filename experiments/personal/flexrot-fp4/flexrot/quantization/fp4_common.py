from __future__ import annotations

from dataclasses import dataclass

import torch


E2M1_CODEBOOK = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


@dataclass(frozen=True)
class FP4QuantizationResult:
    """Fake-quant 输出及审计所需的块 scale、码字和异常掩码。"""

    dequantized: torch.Tensor
    scales: torch.Tensor
    raw_scales: torch.Tensor
    element_codes: torch.Tensor
    scale_codes: torch.Tensor | None
    global_scale: torch.Tensor | None
    zero_block_mask: torch.Tensor
    saturation_mask: torch.Tensor


def reshape_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """把最后一维切成固定 FP4 块，同时保留前导样本维。"""

    if x.ndim == 0 or x.shape[-1] % block_size:
        raise ValueError(f"最后一维必须能被 FP4 block_size={block_size} 整除")
    return x.float().reshape(-1, x.shape[-1] // block_size, block_size)


def nearest_e2m1(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按 round-to-nearest, ties-to-even 映射到有符号 E2M1 格点。

    E2M1 的幅值格点为 ``0,.5,1,1.5,2,3,4,6``。在相邻格点正中间
    时选择偶数索引，从而与已验证代码的 ties-to-even 规则一致。
    """

    magnitude = x.abs().clamp_max(6.0)
    boundaries = E2M1_BOUNDARIES.to(x)
    codebook = E2M1_CODEBOOK.to(x)
    indices = torch.bucketize(magnitude.contiguous(), boundaries, right=False)
    boundary_index = indices.clamp_max(boundaries.numel() - 1)
    at_midpoint = (indices < boundaries.numel()) & (
        magnitude == boundaries[boundary_index]
    )
    indices = indices + (at_midpoint & indices.remainder(2).eq(1)).to(indices.dtype)
    quantized_magnitude = codebook[indices]
    quantized = torch.where(x < 0, -quantized_magnitude, quantized_magnitude)
    signed_codes = torch.where(x < 0, -indices, indices).to(torch.int8)
    return quantized, signed_codes


def telemetry_shape(x: torch.Tensor, block_size: int) -> tuple[int, ...]:
    return (*x.shape[:-1], x.shape[-1] // block_size)
