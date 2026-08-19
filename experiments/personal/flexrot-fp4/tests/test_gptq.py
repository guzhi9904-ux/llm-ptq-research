"""检查 Static ActOrder 恢复物理顺序以及 FP4-aware GPTQ 的基本可运行性。"""

import torch

from flexrot.gptq.gptq import build_hessian, gptq_quantize, prepare_gptq_factor
from flexrot.gptq.mse_grid import prepare_format_projector


def test_gptq_returns_original_weight_layout() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(8, 32, generator=generator)
    activations = torch.randn(128, 32, generator=generator)
    hessian = build_hessian(activations)
    projector = prepare_format_projector(weight, "mxfp4", mse_grid=True, scale_search_iters=8)
    factor = prepare_gptq_factor(weight, hessian, use_static_actorder=True, percdamp=0.01)
    result = gptq_quantize(weight, projector, factor, traversal_chunk=16)
    assert result.quantized_weight.shape == weight.shape
    assert torch.isfinite(result.quantized_weight).all()
    assert sorted(factor.permutation.tolist()) == list(range(weight.shape[1]))
