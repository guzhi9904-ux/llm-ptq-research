"""FP4 基线配置和执行路径测试。"""

from __future__ import annotations

import torch

from ptq_fp4_reference.baselines import FP4BaselineConfig, quantize_fp4_baseline


def test_profiles_keep_baseline_and_mr_gptq_settings_separate() -> None:
    rtn = FP4BaselineConfig("rtn", "mxfp4").as_quantizer_config()
    rotation = FP4BaselineConfig("rotation-rtn", "nvfp4").as_quantizer_config()
    gptq = FP4BaselineConfig("gptq", "mxfp4").as_quantizer_config()

    assert not rtn.use_rotation and not rtn.use_gptq
    assert rotation.use_rotation and not rotation.use_gptq
    assert rotation.hadamard_group_size == 16
    assert not gptq.use_rotation and gptq.use_gptq
    assert not gptq.static_act_order
    assert gptq.relative_damp == 1e-2
    assert {rtn.scale_strategy, rotation.scale_strategy, gptq.scale_strategy} == {"minmax"}


def test_all_baselines_run_for_both_formats() -> None:
    torch.manual_seed(11)
    weight = torch.randn(6, 32)
    activations = torch.randn(48, 32)
    for format_name in ("mxfp4", "nvfp4"):
        for name in ("rtn", "rotation-rtn", "gptq"):
            result = quantize_fp4_baseline(
                weight,
                activations,
                FP4BaselineConfig(name, format_name, update_block_size=16),
            )
            assert result.quantized_weight.shape == weight.shape
            assert torch.isfinite(result.quantized_weight).all()
