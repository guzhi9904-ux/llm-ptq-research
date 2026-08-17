"""根据论文公式独立编写的 MR-GPTQ 数值参考流程。

实现关注 fake-quant 精度语义：先固定原始 group grid，再可选 block Hadamard、
static ActOrder 和 GPTQ 二阶误差补偿。没有复制 FP-Quant 仓库源码，也不包含
QuTLASS packing 或 Blackwell kernel。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .formats import (
    GroupScalePlan,
    MX_SPECS,
    MicroscalingSpec,
    build_group_scale_plan,
    nearest_float_value,
)
from .hadamard import block_hadamard


@dataclass(frozen=True)
class MRGPTQConfig:
    """单个 Linear weight 的参考量化设置。"""

    format_name: str = "mxfp4"
    use_rotation: bool = True
    hadamard_group_size: int = 128
    use_gptq: bool = True
    static_act_order: bool = True
    scale_strategy: str = "mse"
    fit_mxfp_range: bool = False
    relative_damp: float = 0.01
    update_block_size: int = 128
    mse_search_steps: int = 33


@dataclass
class MRGPTQResult:
    """反量化 weight、固定 scale 和诊断信息。"""

    quantized_weight: Tensor
    rotated_weight: Tensor
    scale_plan: GroupScalePlan
    column_order: Tensor
    hessian: Tensor
    weight_mse: float


def calibration_hessian(activations: Tensor) -> Tensor:
    """用校准 activation 构造输入通道 Gram/Hessian 近似。"""

    if activations.ndim < 2:
        raise ValueError("activation 至少需要 token 和 channel 两个维度")
    flattened = activations.reshape(-1, activations.shape[-1]).float()
    return (2.0 / max(flattened.shape[0], 1)) * flattened.t().matmul(flattened)


def _stable_inverse_cholesky(hessian: Tensor, relative_damp: float) -> Tensor:
    diagonal = torch.diag(hessian)
    damping = relative_damp * diagonal.mean().clamp_min(torch.finfo(hessian.dtype).eps)
    stabilized = hessian + torch.eye(
        hessian.shape[0], device=hessian.device, dtype=hessian.dtype
    ) * damping
    # GPTQ 更新使用 inverse Hessian 的 upper Cholesky factor。
    lower = torch.linalg.cholesky(stabilized)
    inverse = torch.cholesky_inverse(lower)
    return torch.linalg.cholesky(inverse, upper=True)


def _quantize_column(values: Tensor, scales: Tensor, spec: MicroscalingSpec) -> Tensor:
    return nearest_float_value(values / scales, spec.element) * scales


def _gptq_with_fixed_scales(
    weight: Tensor,
    hessian: Tensor,
    plan: GroupScalePlan,
    *,
    static_act_order: bool,
    relative_damp: float,
    update_block_size: int,
) -> tuple[Tensor, Tensor]:
    """在固定 grid 上逐列量化，并把误差传播到尚未量化的列。"""

    width = weight.shape[1]
    order = (
        torch.argsort(torch.diag(hessian), descending=True)
        if static_act_order
        else torch.arange(width, device=weight.device)
    )
    working = weight[:, order].clone()
    fixed_scales = plan.expanded_scales()[:, order].to(working)
    ordered_hessian = hessian[order][:, order]
    inverse_factor = _stable_inverse_cholesky(ordered_hessian, relative_damp)
    quantized = torch.zeros_like(working)

    block_size = max(1, update_block_size)
    for start in range(0, width, block_size):
        end = min(start + block_size, width)
        block = working[:, start:end].clone()
        errors = torch.zeros_like(block)
        for local_index, column in enumerate(range(start, end)):
            current = block[:, local_index]
            quantized_column = _quantize_column(
                current, fixed_scales[:, column], plan.spec
            )
            quantized[:, column] = quantized_column
            pivot = inverse_factor[column, column].clamp_min(
                torch.finfo(inverse_factor.dtype).eps
            )
            normalized_error = (current - quantized_column) / pivot
            errors[:, local_index] = normalized_error
            block[:, local_index:] -= normalized_error.unsqueeze(1) * inverse_factor[
                column, column:end
            ].unsqueeze(0)
        if end < width:
            working[:, end:] -= errors.matmul(inverse_factor[start:end, end:])

    restored = torch.empty_like(quantized)
    restored[:, order] = quantized
    return restored, order


def _prepare(
    weight: Tensor, activations: Tensor, config: MRGPTQConfig
) -> tuple[Tensor, Tensor, MicroscalingSpec, GroupScalePlan, Tensor]:
    if weight.ndim != 2 or weight.shape[1] != activations.shape[-1]:
        raise ValueError("weight 形状必须为 [out,in]，activation 最后一维必须等于 in")
    if config.format_name not in {"mxfp4", "nvfp4"}:
        raise ValueError("MR-GPTQ 参考实现当前只支持 mxfp4 或 nvfp4")
    spec = MX_SPECS[config.format_name]
    working_weight = weight.float().clone()
    working_activations = activations.float().clone()
    if config.use_rotation:
        working_weight = block_hadamard(working_weight, config.hadamard_group_size)
        working_activations = block_hadamard(
            working_activations, config.hadamard_group_size
        )
    plan = build_group_scale_plan(
        working_weight,
        spec,
        strategy=config.scale_strategy,
        mse_steps=config.mse_search_steps,
        fit_mxfp_range=config.fit_mxfp_range,
    )
    hessian = calibration_hessian(working_activations)
    return working_weight, working_activations, spec, plan, hessian


def quantize_rtn(
    weight: Tensor, activations: Tensor, config: MRGPTQConfig
) -> MRGPTQResult:
    """使用同一格式、rotation 和 scale 规则构造公平 RTN 基线。"""

    working_weight, _, _, plan, hessian = _prepare(weight, activations, config)
    quantized = plan.quantize(working_weight)
    order = torch.arange(weight.shape[1], device=weight.device)
    return MRGPTQResult(
        quantized_weight=quantized.to(weight.dtype),
        rotated_weight=working_weight.to(weight.dtype),
        scale_plan=plan,
        column_order=order,
        hessian=hessian,
        weight_mse=float((quantized - working_weight).square().mean()),
    )


def quantize_mr_gptq(
    weight: Tensor, activations: Tensor, config: MRGPTQConfig
) -> MRGPTQResult:
    """执行 MR-GPTQ；use_gptq=False 时退化为同配置 RTN。"""

    if not config.use_gptq:
        return quantize_rtn(weight, activations, config)
    working_weight, _, _, plan, hessian = _prepare(weight, activations, config)
    quantized, order = _gptq_with_fixed_scales(
        working_weight,
        hessian,
        plan,
        static_act_order=config.static_act_order,
        relative_damp=config.relative_damp,
        update_block_size=config.update_block_size,
    )
    return MRGPTQResult(
        quantized_weight=quantized.to(weight.dtype),
        rotated_weight=working_weight.to(weight.dtype),
        scale_plan=plan,
        column_order=order,
        hessian=hessian,
        weight_mse=float((quantized - working_weight).square().mean()),
    )


def mr_gptq_fake_linear(
    inputs: Tensor,
    result: MRGPTQResult,
    config: MRGPTQConfig,
    bias: Tensor | None = None,
) -> Tensor:
    """用动态 activation fake quant 评估已量化 weight 的线性输出。"""

    from .formats import microscale_fake_quant

    transformed = (
        block_hadamard(inputs, config.hadamard_group_size)
        if config.use_rotation
        else inputs
    )
    quantized_inputs = microscale_fake_quant(transformed, MX_SPECS[config.format_name])
    return torch.nn.functional.linear(quantized_inputs, result.quantized_weight, bias)
