from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from .gptq import build_hessian, gptq_quantize, prepare_gptq_factor
from .mse_grid import prepare_format_projector


@dataclass(frozen=True)
class MRGPTQConfig:
    """MR-GPTQ 中与模型无关、应由 YAML 显式传入的算法参数。"""

    damping: float = 0.01
    traversal_chunk: int = 128
    static_actorder: bool = True
    mse_grid: bool = True
    mse_grid_iters: int = 100
    max_scale_shrink_factor: float = 0.80
    error_norm: float = 2.4


@torch.no_grad()
def quantize_current_linear(
    weight: torch.Tensor,
    calibration_activations: torch.Tensor,
    *,
    format_name: str,
    config: MRGPTQConfig,
) -> torch.Tensor:
    """量化当前 Linear；调用者负责先对 X/W 成对旋转。

    MR-GPTQ pipeline：输入 BF16 模型 → 构建 rotation → transform X/W →
    activation calibration → Static ActOrder → FP4-aware GPTQ → MSE-grid →
    sequential deployment calibration → W4A4 forward → WikiText-2 PPL。

    本函数覆盖中间的单 Linear 核心；模型适配器按层调用，并在当前层完成后
    将量化权重移回 CPU。这样 GPU 上不会同时存在 BF16 copy、rotation copy、
    quantized copy、全部 Hessian 和全部 activation cache。
    """

    hessian = build_hessian(calibration_activations)
    projector = prepare_format_projector(
        weight,
        format_name,
        mse_grid=config.mse_grid,
        scale_search_iters=config.mse_grid_iters,
        max_scale_shrink_factor=config.max_scale_shrink_factor,
        error_norm=config.error_norm,
    )
    factor = prepare_gptq_factor(
        weight,
        hessian,
        use_static_actorder=config.static_actorder,
        percdamp=config.damping,
    )
    result = gptq_quantize(
        weight,
        projector,
        factor,
        compensate=True,
        traversal_chunk=config.traversal_chunk,
    )
    quantized = result.quantized_weight
    # 这里主动释放 Hessian，避免 7B/8B 模型在单 4090 上累计显存。
    del hessian, projector, factor, result
    gc.collect()
    if weight.device.type == "cuda":
        torch.cuda.empty_cache()
    return quantized


def release_layer_temporaries(*objects: object) -> None:
    """逐层 pipeline 的统一显存回收点；activation cache 应保存在 CPU/磁盘。"""

    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
