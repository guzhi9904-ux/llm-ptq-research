from __future__ import annotations

import torch

from .fp4_common import (
    FP4QuantizationResult,
    nearest_e2m1,
    reshape_blocks,
    telemetry_shape,
)


MXFP4_BLOCK_SIZE = 32
E2M1_MAX = 6.0
MXFP4_SCALE_MODES = frozenset({"legacy", "paper"})


def _e8m0_scales(raw_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 raw scale 向上取整为 E8M0 所表示的二次幂。

    E8M0 没有尾数，因而 scale 只能是 ``2**exponent``。向上取整
    ``ceil(log2(raw))`` 可确保块内最大值不因 scale 过小而必然溢出；指数
    限制在 [-127,127]。沿用已验证实现，全零块记 exponent=-1、scale=0.5，
    但元素码字全为零，所以反量化语义仍是精确零块。
    """

    nonzero = raw_scales > 0
    exponent = torch.ceil(
        torch.log2(raw_scales.clamp_min(torch.finfo(torch.float32).tiny))
    ).clamp(-127, 127).to(torch.int16)
    exponent = torch.where(nonzero, exponent, torch.full_like(exponent, -1))
    scales = torch.ldexp(torch.ones_like(raw_scales), exponent.to(torch.int32))
    return scales, exponent


def _paper_e8m0_scales(raw_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """复现 FP-Quant 的 MXFP scale-fitting 有效尺度。

    官方实现先对 ``amax`` 取向下 E8M0 指数并减去 E2M1 的最大指数 2，
    再除以固定 ``FP4_SCALE=3/4``。存储的仍是 E8M0 指数；这里返回包含
    固定 4/3 因子的反量化有效尺度，等价于官方 fake-quant 路径。
    """

    amax = raw_scales * E2M1_MAX
    exponent = (
        torch.floor(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny)))
        .clamp(-127, 127)
        .sub(2)
        .to(torch.int16)
    )
    encoded = torch.ldexp(torch.ones_like(raw_scales), exponent.to(torch.int32))
    effective = encoded / 0.75
    return effective, exponent


def mxfp4_scales(
    raw_scales: torch.Tensor, *, scale_mode: str = "paper"
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale_mode == "paper":
        return _paper_e8m0_scales(raw_scales)
    if scale_mode == "legacy":
        return _e8m0_scales(raw_scales)
    raise ValueError(f"未知 MXFP4 scale_mode={scale_mode!r}；应为 paper 或 legacy")


def fake_quant_mxfp4(
    x: torch.Tensor, *, scale_mode: str = "paper"
) -> FP4QuantizationResult:
    """执行 MXFP4 E2M1/block32/E8M0 fake quant。

    MXFP4 与 NVFP4 共享 E2M1 元素格点，最大差异是 scale：MXFP4 每个
    32 元素块独立使用二次幂 E8M0，不存在 tensor-wide global scale；
    NVFP4 则是 16 元素块的 E4M3 local scale 再乘一个 FP32 global scale。
    """

    blocks = reshape_blocks(x, MXFP4_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    zero_blocks = amax.eq(0)
    raw_scales = amax / E2M1_MAX
    scales, scale_codes = mxfp4_scales(raw_scales, scale_mode=scale_mode)
    normalized = blocks / scales
    quantized, element_codes = nearest_e2m1(normalized)
    output = (quantized * scales).reshape_as(x)
    shape = telemetry_shape(x, MXFP4_BLOCK_SIZE)
    return FP4QuantizationResult(
        dequantized=output,
        scales=scales.reshape(shape),
        raw_scales=raw_scales.reshape(shape),
        element_codes=element_codes.reshape_as(x),
        scale_codes=scale_codes.reshape(shape),
        global_scale=None,
        zero_block_mask=zero_blocks.reshape(shape),
        saturation_mask=(normalized.abs() > E2M1_MAX).reshape_as(x),
    )
