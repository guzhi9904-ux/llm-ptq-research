from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from flexrot.quantization.mxfp4 import fake_quant_mxfp4
from flexrot.quantization.nvfp4 import fake_quant_nvfp4
from flexrot.rotation.transforms import apply_rotation


SUPPORTED_LINEAR_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def discover_linear_modules(model: nn.Module) -> dict[str, nn.Linear]:
    """发现 Qwen2/Qwen3/Llama decoder 中受支持的七类 Linear。"""

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1] in SUPPORTED_LINEAR_NAMES
    }


class FP4Linear(nn.Module):
    """保存预量化权重，并在 forward 对旋转后的 activation 做 FP4 fake quant。"""

    def __init__(
        self,
        source: nn.Linear,
        quantized_weight: torch.Tensor,
        *,
        format_name: str,
        block_size: int,
        theta_deg: float,
        activation_global_scale: float | None,
        quantize_activation: bool,
        mxfp_scale_mode: str = "paper",
    ) -> None:
        super().__init__()
        self.format_name = format_name
        self.block_size = block_size
        self.theta_deg = theta_deg
        self.activation_global_scale = activation_global_scale
        self.quantize_activation = quantize_activation
        self.mxfp_scale_mode = mxfp_scale_mode
        self.weight = nn.Parameter(
            quantized_weight.to(source.weight), requires_grad=False
        )
        self.bias = source.bias
        self.in_features = source.in_features
        self.out_features = source.out_features

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # 实验协议把 0° 定义为“不旋转”；F(0) 的符号对角仅是数学族端点，
        # 不作为正式 arm 的运行实现。
        rotated = (
            values
            if self.theta_deg == 0.0
            else apply_rotation(
                # 与已验证生产路径一致：旋转矩阵乘在 FP32 中完成，再转回
                # Linear 权重 dtype。BF16 直接旋转会改变弱角度的量化景观。
                values.float(),
                block_size=self.block_size,
                theta_deg=self.theta_deg,
            )
        ).to(self.weight.dtype)
        if self.quantize_activation:
            if self.format_name == "nvfp4":
                rotated = fake_quant_nvfp4(
                    rotated, global_scale=self.activation_global_scale
                ).dequantized.to(self.weight.dtype)
            else:
                rotated = fake_quant_mxfp4(
                    rotated, scale_mode=self.mxfp_scale_mode
                ).dequantized.to(self.weight.dtype)
        return functional.linear(rotated, self.weight, self.bias)


@dataclass(frozen=True)
class ReplacementReport:
    module_names: tuple[str, ...]
    quantized_weights: int
    quantized_activations: int


@torch.no_grad()
def replace_with_rtn(
    model: nn.Module,
    *,
    format_name: str,
    theta_deg: float,
    activation_global_scales: dict[str, float] | None = None,
    quantize_activation: bool = True,
    mxfp_scale_mode: str = "paper",
) -> ReplacementReport:
    """用 FlexRot+RTN W4A4 Linear 原位替换受支持模块，不复制整个模型。"""

    if format_name not in {"nvfp4", "mxfp4"}:
        raise ValueError("format_name 必须为 nvfp4 或 mxfp4")
    block_size = 16 if format_name == "nvfp4" else 32
    scales = activation_global_scales or {}
    names = tuple(discover_linear_modules(model))
    if quantize_activation and format_name == "nvfp4":
        missing = sorted(set(names) - set(scales))
        if missing:
            raise ValueError(f"缺少 NVFP4 activation global scale：{missing[:3]}")
    for name in names:
        linear = model.get_submodule(name)
        rotated_weight = (
            linear.weight.detach()
            if theta_deg == 0.0
            else apply_rotation(
                linear.weight.detach().float(),
                block_size=block_size,
                theta_deg=theta_deg,
            )
        )
        if format_name == "nvfp4":
            quantized_weight = fake_quant_nvfp4(rotated_weight).dequantized
        else:
            quantized_weight = fake_quant_mxfp4(
                rotated_weight, scale_mode=mxfp_scale_mode
            ).dequantized
        replacement = FP4Linear(
            linear,
            quantized_weight,
            format_name=format_name,
            block_size=block_size,
            theta_deg=theta_deg,
            activation_global_scale=scales.get(name),
            quantize_activation=quantize_activation,
            mxfp_scale_mode=mxfp_scale_mode,
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)
    return ReplacementReport(
        module_names=names,
        quantized_weights=len(names),
        quantized_activations=len(names) if quantize_activation else 0,
    )
