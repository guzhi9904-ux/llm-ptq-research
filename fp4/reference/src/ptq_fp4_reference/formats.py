"""Microscaling 浮点格式的可读 fake-quant 实现。

这里保存反量化后的 PyTorch 张量，不做 bit packing，也没有模拟 Blackwell
Tensor Core 的逐 bit 行为。量化级别和 block scale 可以单独测试。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class FloatFormatSpec:
    """一个有限低精度浮点元素格式。"""

    name: str
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    max_normal: float

    @property
    def bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits


@dataclass(frozen=True)
class MicroscalingSpec:
    """元素格式、共享 block 和 scale 编码的组合。"""

    name: str
    element: FloatFormatSpec
    group_size: int
    scale_encoding: str


FORMAT_SPECS = {
    "e2m1": FloatFormatSpec("E2M1", 2, 1, 1, 6.0),
    # MixFP4 使用的 E1M2：按论文附录采用 bias=0 和 subnormal，
    # 正数网格为 0, 0.5, 1.0, ..., 3.5。
    "e1m2": FloatFormatSpec("E1M2", 1, 2, 0, 3.5),
    "e3m2": FloatFormatSpec("E3M2", 3, 2, 3, 28.0),
    "e2m3": FloatFormatSpec("E2M3", 2, 3, 1, 7.5),
    "e4m3": FloatFormatSpec("E4M3", 4, 3, 7, 448.0),
    "e5m2": FloatFormatSpec("E5M2", 5, 2, 15, 57344.0),
}

MX_SPECS = {
    "mxfp4": MicroscalingSpec("MXFP4", FORMAT_SPECS["e2m1"], 32, "e8m0"),
    "nvfp4": MicroscalingSpec("NVFP4", FORMAT_SPECS["e2m1"], 16, "e4m3"),
    "mxfp6": MicroscalingSpec("MXFP6", FORMAT_SPECS["e3m2"], 32, "e8m0"),
    "mxfp8": MicroscalingSpec("MXFP8", FORMAT_SPECS["e5m2"], 32, "e8m0"),
}


def positive_levels(spec: FloatFormatSpec, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """枚举非负有限值，并按格式最大有限值截断特殊编码。"""

    values = {0.0}
    mantissa_count = 1 << spec.mantissa_bits
    # exponent field 为 0 时生成 subnormal。
    for mantissa in range(1, mantissa_count):
        fraction = mantissa / mantissa_count
        values.add(fraction * (2.0 ** (1 - spec.exponent_bias)))
    max_exponent_field = (1 << spec.exponent_bits) - 1
    for exponent_field in range(1, max_exponent_field + 1):
        exponent = exponent_field - spec.exponent_bias
        for mantissa in range(mantissa_count):
            value = (1.0 + mantissa / mantissa_count) * (2.0**exponent)
            if value <= spec.max_normal:
                values.add(value)
    return torch.tensor(sorted(values), device=device, dtype=dtype)


def nearest_float_value(values: Tensor, spec: FloatFormatSpec) -> Tensor:
    """把张量舍入到给定低精度浮点网格，tie 时选择较小幅值。"""

    levels = positive_levels(spec, device=values.device, dtype=values.dtype)
    magnitude = values.abs().contiguous()
    upper_index = torch.bucketize(magnitude, levels).clamp(max=levels.numel() - 1)
    lower_index = (upper_index - 1).clamp(min=0)
    lower = levels[lower_index]
    upper = levels[upper_index]
    rounded = torch.where((upper - magnitude) < (magnitude - lower), upper, lower)
    return rounded.copysign(values)


def _reshape_blocks(values: Tensor, group_size: int) -> tuple[Tensor, int]:
    if group_size <= 0:
        raise ValueError("group_size 必须为正整数")
    width = values.shape[-1]
    padding = (-width) % group_size
    if padding:
        values = torch.nn.functional.pad(values, (0, padding))
    return values.reshape(*values.shape[:-1], -1, group_size), padding


def _restore_blocks(blocks: Tensor, padding: int) -> Tensor:
    restored = blocks.flatten(start_dim=-2)
    return restored[..., : restored.shape[-1] - padding] if padding else restored


def mx_shared_scale(blocks: Tensor, element: FloatFormatSpec) -> Tensor:
    """按 MicroMix 论文公式 3 生成 E8M0 风格的 2 的幂 scale。"""

    absmax = blocks.abs().amax(dim=-1, keepdim=True)
    safe = absmax.clamp_min(torch.finfo(blocks.dtype).tiny)
    exponent = torch.floor(torch.log2(safe)) - element.exponent_bias
    scale = torch.pow(torch.tensor(2.0, device=blocks.device, dtype=blocks.dtype), exponent)
    return torch.where(absmax == 0, torch.ones_like(scale), scale)


def microscale_fake_quant(values: Tensor, spec: MicroscalingSpec) -> Tensor:
    """按最后一维分 block 执行 MX fake quant 并返回反量化张量。"""

    if not values.is_floating_point():
        raise TypeError("fake quant 输入必须是浮点张量")
    original_dtype = values.dtype
    working = values.float() if values.dtype in {torch.float16, torch.bfloat16} else values
    blocks, padding = _reshape_blocks(working, spec.group_size)
    scale = mx_shared_scale(blocks, spec.element)
    quantized = nearest_float_value(blocks / scale, spec.element) * scale
    return _restore_blocks(quantized, padding).to(original_dtype)


@dataclass
class GroupScalePlan:
    """为 weight 固定的逐行、逐组 scale。"""

    spec: MicroscalingSpec
    scales: Tensor
    width: int

    def expanded_scales(self) -> Tensor:
        expanded = self.scales.repeat_interleave(self.spec.group_size, dim=-1)
        return expanded[..., : self.width]

    def quantize(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.width:
            raise ValueError("输入宽度与 scale plan 不一致")
        scales = self.expanded_scales().to(device=values.device, dtype=values.dtype)
        return nearest_float_value(values / scales, self.spec.element) * scales


def _raw_group_scales(blocks: Tensor, spec: MicroscalingSpec) -> Tensor:
    absmax = blocks.abs().amax(dim=-1)
    return (absmax / spec.element.max_normal).clamp_min(torch.finfo(blocks.dtype).tiny)


def _encode_e8m0(scales: Tensor) -> Tensor:
    """使用 MR-GPTQ 附录公式 1 的 4/3 重缩放 E8M0 近似。"""

    exponent = torch.round(torch.log2(scales)).clamp(-128, 127)
    return (4.0 / 3.0) * torch.pow(
        torch.tensor(2.0, device=scales.device, dtype=scales.dtype), exponent
    )


def fit_e8m0_scales(scales: Tensor) -> Tensor:
    """按 MR-GPTQ 附录 H 把有效 log2 范围映射到 256 个 level。"""

    log_scales = torch.log2(scales)
    minimum = log_scales.amin()
    maximum = log_scales.amax()
    span = maximum - minimum
    if float(span.abs()) < 1e-12:
        return scales.clone()
    code = torch.round(255.0 * (log_scales - minimum) / span).clamp(0, 255)
    fitted_log = minimum + code * span / 255.0
    return torch.pow(torch.tensor(2.0, device=scales.device, dtype=scales.dtype), fitted_log)


def _encode_e4m3_group_scales(scales: Tensor) -> Tensor:
    """用一个 tensor scale 加 E4M3 group scale 近似 NVFP4 的两级 scale。"""

    maximum = scales.amax().clamp_min(torch.finfo(scales.dtype).tiny)
    tensor_scale = maximum / FORMAT_SPECS["e4m3"].max_normal
    encoded = nearest_float_value(scales / tensor_scale, FORMAT_SPECS["e4m3"])
    return (encoded * tensor_scale).clamp_min(torch.finfo(scales.dtype).tiny)


def _mse_scales(blocks: Tensor, spec: MicroscalingSpec, steps: int) -> Tensor:
    """逐组搜索 clipping scale；实现清晰优先，不针对大模型吞吐优化。"""

    base = _raw_group_scales(blocks, spec)
    best = base.clone()
    best_error = torch.full_like(base, math.inf)
    for multiplier in torch.linspace(0.55, 1.05, steps, device=blocks.device, dtype=blocks.dtype):
        candidate = base * multiplier
        quantized = nearest_float_value(blocks / candidate.unsqueeze(-1), spec.element)
        reconstructed = quantized * candidate.unsqueeze(-1)
        error = (reconstructed - blocks).square().mean(dim=-1)
        take = error < best_error
        best_error = torch.where(take, error, best_error)
        best = torch.where(take, candidate, best)
    return best


def build_group_scale_plan(
    values: Tensor,
    spec: MicroscalingSpec,
    *,
    strategy: str = "minmax",
    mse_steps: int = 33,
    fit_mxfp_range: bool = False,
) -> GroupScalePlan:
    """在原始列顺序上固定 scale，供 RTN 或 static ActOrder GPTQ 使用。"""

    working = values.float() if values.dtype in {torch.float16, torch.bfloat16} else values
    blocks, _ = _reshape_blocks(working, spec.group_size)
    if strategy == "minmax":
        scales = _raw_group_scales(blocks, spec)
    elif strategy == "mse":
        scales = _mse_scales(blocks, spec, mse_steps)
    else:
        raise ValueError(f"未知 scale strategy：{strategy}")

    if spec.scale_encoding == "e8m0":
        scales = fit_e8m0_scales(scales) if fit_mxfp_range else _encode_e8m0(scales)
    elif spec.scale_encoding == "e4m3":
        scales = _encode_e4m3_group_scales(scales)
    else:
        raise ValueError(f"未知 scale encoding：{spec.scale_encoding}")
    return GroupScalePlan(spec=spec, scales=scales, width=values.shape[-1])
