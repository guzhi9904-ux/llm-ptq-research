"""根据 MicroMix 论文公式独立实现的离线 channel 分区参考流程。

该文件实现 activation 统计、阈值比例、静态 permutation 以及 MXFP4/6/8
fake quant。它不包含原仓库的 CUDA 代码，也不声称复现 Blackwell kernel 延迟。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .formats import MX_SPECS, MicroscalingSpec, microscale_fake_quant


@dataclass(frozen=True)
class PrecisionPartition:
    bits: int
    start: int
    end: int

    @property
    def channels(self) -> int:
        return self.end - self.start


@dataclass
class MicroMixPlan:
    """一个 Linear 层离线固定的 channel 顺序和精度边界。"""

    permutation: Tensor
    channel_scores: Tensor
    partitions: tuple[PrecisionPartition, ...]
    threshold_proportions: dict[int, float]

    @property
    def channels(self) -> int:
        return int(self.permutation.numel())

    @property
    def average_element_bits(self) -> float:
        total = sum(item.bits * item.channels for item in self.partitions)
        return total / max(self.channels, 1)

    @property
    def average_storage_bits(self) -> float:
        # 三种 MX 格式均为每 32 个元素共享一个 8-bit E8M0 scale。
        return self.average_element_bits + 8.0 / 32.0


def channel_absolute_mean(activations: Tensor) -> Tensor:
    """论文公式 9：对除 channel 外的维度计算绝对值均值。"""

    if activations.ndim < 2:
        raise ValueError("activation 至少需要 token 和 channel 两个维度")
    flattened = activations.reshape(-1, activations.shape[-1]).float()
    return flattened.abs().mean(dim=0)


def quantization_threshold(maximum: Tensor, bits: int) -> Tensor:
    """论文公式 6/21：把 MX 误差上界限制在 INT8 上界内。"""

    if bits == 4:
        exponent_bias, qmax = 1, 6.0  # MXFP4 E2M1
    elif bits == 6:
        exponent_bias, qmax = 3, 28.0  # MXFP6 E3M2
    else:
        raise ValueError("阈值只对 MXFP4 和 MXFP6 定义")
    coefficient = (2.0**exponent_bias) * (2.0 ** (bits - 1)) / qmax
    return coefficient * maximum / 254.0


def estimate_precision_proportions(activations: Tensor) -> dict[int, float]:
    """逐 token 使用论文阈值，再对校准样本汇总 4/6/8-bit 元素比例。

    论文给出逐层 p4/p6/p8 与离线固定策略，但未规定所有实现细节。这里采用
    对校准元素比例求均值的透明规则，并在 README 中标为参考选择。
    """

    flattened = activations.reshape(-1, activations.shape[-1]).float().abs()
    maximum = flattened.amax(dim=-1, keepdim=True)
    threshold4 = quantization_threshold(maximum, 4)
    threshold6 = quantization_threshold(maximum, 6)
    count = max(flattened.numel(), 1)
    p4 = float((flattened <= threshold4).sum()) / count
    p6 = float(((flattened > threshold4) & (flattened <= threshold6)).sum()) / count
    p8 = max(0.0, 1.0 - p4 - p6)
    return {4: p4, 6: p6, 8: p8}


def _integer_counts(channels: int, proportions: dict[int, float], alignment: int) -> dict[int, int]:
    raw = {bits: proportions[bits] * channels for bits in (4, 6, 8)}
    base = {bits: int(raw[bits]) for bits in (4, 6, 8)}
    remaining = channels - sum(base.values())
    order = sorted((4, 6, 8), key=lambda bits: raw[bits] - base[bits], reverse=True)
    for index in range(remaining):
        base[order[index % len(order)]] += 1
    if alignment > 1:
        # 低精度区向下对齐，余数交给最高精度，保证不降低保守性。
        base[4] = (base[4] // alignment) * alignment
        base[6] = (base[6] // alignment) * alignment
        base[8] = channels - base[4] - base[6]
    return base


def build_micromix_plan(activations: Tensor, *, channel_alignment: int = 1) -> MicroMixPlan:
    """按 activation 绝对均值升序排列，并固定每层 4/6/8-bit channel 数。"""

    if channel_alignment <= 0:
        raise ValueError("channel_alignment 必须为正整数")
    scores = channel_absolute_mean(activations)
    permutation = torch.argsort(scores, descending=False)
    proportions = estimate_precision_proportions(activations)
    counts = _integer_counts(scores.numel(), proportions, channel_alignment)
    boundary4 = counts[4]
    boundary6 = boundary4 + counts[6]
    partitions = (
        PrecisionPartition(4, 0, boundary4),
        PrecisionPartition(6, boundary4, boundary6),
        PrecisionPartition(8, boundary6, scores.numel()),
    )
    return MicroMixPlan(
        permutation=permutation,
        channel_scores=scores,
        partitions=partitions,
        threshold_proportions=proportions,
    )


def _spec_for_bits(bits: int, fp8_variant: str) -> MicroscalingSpec:
    if bits == 4:
        return MX_SPECS["mxfp4"]
    if bits == 6:
        return MX_SPECS["mxfp6"]
    if bits == 8 and fp8_variant == "e5m2":
        return MX_SPECS["mxfp8"]
    if bits == 8 and fp8_variant == "e4m3":
        from .formats import FORMAT_SPECS

        return MicroscalingSpec("MXFP8-E4M3", FORMAT_SPECS["e4m3"], 32, "e8m0")
    raise ValueError(f"不支持的精度组合：bits={bits}, fp8_variant={fp8_variant}")


def micromix_fake_quant(
    reordered_values: Tensor, plan: MicroMixPlan, *, fp8_variant: str = "e5m2"
) -> Tensor:
    """对已重排张量的三个连续 channel 区间分别执行 MX fake quant。"""

    if reordered_values.shape[-1] != plan.channels:
        raise ValueError("输入 channel 数与 MicroMix plan 不一致")
    outputs: list[Tensor] = []
    for partition in plan.partitions:
        segment = reordered_values[..., partition.start : partition.end]
        if partition.channels:
            segment = microscale_fake_quant(
                segment, _spec_for_bits(partition.bits, fp8_variant)
            )
        outputs.append(segment)
    return torch.cat(outputs, dim=-1)


def micromix_fake_linear(
    inputs: Tensor,
    weight: Tensor,
    plan: MicroMixPlan,
    *,
    bias: Tensor | None = None,
    fp8_variant: str = "e5m2",
) -> Tensor:
    """重排并以相同精度量化 activation/weight 对应 channel。"""

    if weight.ndim != 2 or weight.shape[-1] != plan.channels:
        raise ValueError("weight 必须为 [out,in]，且 in 与 plan 一致")
    permutation = plan.permutation.to(inputs.device)
    reordered_inputs = inputs[..., permutation]
    reordered_weight = weight[:, permutation.to(weight.device)]
    quantized_inputs = micromix_fake_quant(
        reordered_inputs, plan, fp8_variant=fp8_variant
    )
    quantized_weight = micromix_fake_quant(
        reordered_weight, plan, fp8_variant=fp8_variant
    )
    return torch.nn.functional.linear(quantized_inputs, quantized_weight, bias)
