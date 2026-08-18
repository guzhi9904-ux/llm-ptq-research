"""论文实验中使用的 FP4 基线配置。

这里把 RTN、Hadamard + RTN 和标准 GPTQ 分成三个明确入口。三者都使用
absmax/minmax scale；MSE scale、static ActOrder 和 GPTQ 前旋转属于 MR-GPTQ，
不应悄悄混进普通 GPTQ 基线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor

from .formats import MX_SPECS
from .mr_gptq import MRGPTQConfig, MRGPTQResult, quantize_mr_gptq


BaselineName = Literal["rtn", "rotation-rtn", "gptq"]


@dataclass(frozen=True)
class FP4BaselineConfig:
    """一条基础实验所需的参数。"""

    name: BaselineName
    format_name: str = "mxfp4"
    hadamard_group_size: int | None = None
    relative_damp: float = 1e-2
    update_block_size: int = 128

    def as_quantizer_config(self) -> MRGPTQConfig:
        if self.format_name not in {"mxfp4", "nvfp4"}:
            raise ValueError("baseline format 只能是 mxfp4 或 nvfp4")
        if self.name not in {"rtn", "rotation-rtn", "gptq"}:
            raise ValueError(f"未知 FP4 baseline：{self.name}")

        rotation = self.name == "rotation-rtn"
        rotation_size = self.hadamard_group_size or MX_SPECS[self.format_name].group_size
        return MRGPTQConfig(
            format_name=self.format_name,
            use_rotation=rotation,
            hadamard_group_size=rotation_size,
            use_gptq=self.name == "gptq",
            # MR-GPTQ 论文中的普通 GPTQ 使用 standard/default order。
            static_act_order=False,
            # 三条 baseline 都使用 absmax scale；实现中对应 minmax。
            scale_strategy="minmax",
            fit_mxfp_range=False,
            relative_damp=self.relative_damp,
            update_block_size=self.update_block_size,
        )


def quantize_fp4_baseline(
    weight: Tensor,
    activations: Tensor,
    config: FP4BaselineConfig,
) -> MRGPTQResult:
    """在 MXFP4 或 NVFP4 网格上运行一条基础实验。"""

    return quantize_mr_gptq(weight, activations, config.as_quantizer_config())
