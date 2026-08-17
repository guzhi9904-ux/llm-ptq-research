"""MR-GPTQ 核心张量流程测试。"""

from __future__ import annotations

import torch

from ptq_fp4_reference.mr_gptq import (
    MRGPTQConfig,
    mr_gptq_fake_linear,
    quantize_mr_gptq,
    quantize_rtn,
)


def make_problem() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    weight = torch.randn(9, 128)
    activations = torch.randn(64, 128)
    # 制造不同 channel 的 Hessian 对角线，便于验证 static ActOrder。
    activations[:, 7] *= 12.0
    activations[:, 39] *= 5.0
    return weight, activations


def test_static_act_order_follows_hessian_diagonal() -> None:
    weight, activations = make_problem()
    # 单独验证 ActOrder 时关闭 rotation，避免 Hadamard 有意分散第 7 通道能量。
    config = MRGPTQConfig(
        use_rotation=False, mse_search_steps=5, update_block_size=32
    )
    result = quantize_mr_gptq(weight, activations, config)
    expected = torch.argsort(torch.diag(result.hessian), descending=True)
    assert torch.equal(result.column_order, expected)
    assert result.column_order[0].item() == 7


def test_gptq_result_is_deterministic_and_finite() -> None:
    weight, activations = make_problem()
    config = MRGPTQConfig(mse_search_steps=5, update_block_size=32)
    first = quantize_mr_gptq(weight, activations, config)
    second = quantize_mr_gptq(weight, activations, config)
    torch.testing.assert_close(first.quantized_weight, second.quantized_weight)
    assert torch.isfinite(first.quantized_weight).all()
    assert first.weight_mse >= 0.0


def test_disabled_gptq_matches_rtn() -> None:
    weight, activations = make_problem()
    config = MRGPTQConfig(use_gptq=False, mse_search_steps=5)
    via_dispatch = quantize_mr_gptq(weight, activations, config)
    direct = quantize_rtn(weight, activations, config)
    torch.testing.assert_close(via_dispatch.quantized_weight, direct.quantized_weight)


def test_fake_linear_has_expected_shape() -> None:
    weight, activations = make_problem()
    config = MRGPTQConfig(mse_search_steps=5, update_block_size=32)
    result = quantize_mr_gptq(weight, activations, config)
    output = mr_gptq_fake_linear(activations[:6], result, config)
    assert output.shape == (6, weight.shape[0])
    assert torch.isfinite(output).all()


def test_gptq_reduces_calibration_output_mse_against_matched_rtn() -> None:
    weight, activations = make_problem()
    common = {"mse_search_steps": 5, "update_block_size": 32}
    rtn_config = MRGPTQConfig(use_gptq=False, **common)
    gptq_config = MRGPTQConfig(use_gptq=True, **common)

    rtn_result = quantize_rtn(weight, activations, rtn_config)
    gptq_result = quantize_mr_gptq(weight, activations, gptq_config)
    target = activations @ weight.t()
    rtn_mse = (
        mr_gptq_fake_linear(activations, rtn_result, rtn_config) - target
    ).square().mean()
    gptq_mse = (
        mr_gptq_fake_linear(activations, gptq_result, gptq_config) - target
    ).square().mean()

    # 固定随机问题上的回归检查：二阶误差补偿应优于同量化网格的 RTN。
    assert gptq_mse < rtn_mse
