from __future__ import annotations

from collections.abc import Iterable

import torch

from .fp4_common import (
    FP4QuantizationResult,
    nearest_e2m1,
    reshape_blocks,
    telemetry_shape,
)


NVFP4_BLOCK_SIZE = 16
E4M3_MAX = 448.0
E2M1_MAX = 6.0


def calibrate_nvfp4_global_scale(
    batches: torch.Tensor | Iterable[torch.Tensor],
) -> float:
    """用校准数据计算 NVFP4 的 FP32 global scale。

    每个 16 元素块先需要 ``raw_scale=amax/6``，随后 raw scale 会编码成
    E4M3 local scale。单个 E4M3 只能表示到 448，因此 global scale 取
    ``max(raw_scale)/448``。激活极值依赖数据分布，不能只从模型权重推断，
    所以 activation global scale 必须由与测试集分离的 calibration 数据求得。
    """

    stream = (batches,) if isinstance(batches, torch.Tensor) else batches
    maximum = 0.0
    seen = False
    for values in stream:
        seen = True
        blocks = reshape_blocks(values, NVFP4_BLOCK_SIZE)
        maximum = max(maximum, float(blocks.abs().amax().item()))
    if not seen:
        raise ValueError("校准数据不能为空")
    return maximum / (E2M1_MAX * E4M3_MAX) if maximum else 1.0


def _e4m3_scales(
    raw_scales: torch.Tensor, global_scale: float | torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = raw_scales.max()
    if global_scale is None:
        resolved = torch.where(
            maximum > 0, maximum / E4M3_MAX, torch.ones_like(maximum)
        )
    else:
        resolved = torch.as_tensor(
            global_scale, dtype=torch.float32, device=raw_scales.device
        ).reshape(())
        if not bool(torch.isfinite(resolved)) or float(resolved) <= 0:
            raise ValueError("NVFP4 global_scale 必须为有限正数")
    normalized = torch.where(
        raw_scales > 0, raw_scales / resolved, torch.zeros_like(raw_scales)
    )
    encoded = normalized.clamp_max(E4M3_MAX).to(torch.float8_e4m3fn)
    scales = encoded.float() * resolved
    return scales, encoded.view(torch.uint8), resolved


def fake_quant_nvfp4(
    x: torch.Tensor,
    *,
    global_scale: float | torch.Tensor | None = None,
) -> FP4QuantizationResult:
    """执行 NVFP4 E2M1/block16/E4M3+FP32-global fake quant。

    权重可令 ``global_scale=None`` 并由当前张量确定；激活应传入独立校准的
    global scale。全零块的 local scale 编码为 0，除法时临时使用 1，最终
    码字和反量化结果仍严格为 0，不会产生 NaN。权重与激活使用同一 E2M1
    舍入流程，差别仅在 activation scale 来自 calibration。
    """

    blocks = reshape_blocks(x, NVFP4_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    zero_blocks = amax.eq(0)
    raw_scales = amax / E2M1_MAX
    scales, scale_codes, resolved_global = _e4m3_scales(raw_scales, global_scale)
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    normalized = blocks / safe_scales
    quantized, element_codes = nearest_e2m1(normalized)
    output = (quantized * scales).reshape_as(x)
    shape = telemetry_shape(x, NVFP4_BLOCK_SIZE)
    return FP4QuantizationResult(
        dequantized=output,
        scales=scales.reshape(shape),
        raw_scales=raw_scales.reshape(shape),
        element_codes=element_codes.reshape_as(x),
        scale_codes=scale_codes.reshape(shape),
        global_scale=resolved_global,
        zero_block_mask=zero_blocks.reshape(shape),
        saturation_mask=(normalized.abs() > E2M1_MAX).reshape_as(x),
    )
