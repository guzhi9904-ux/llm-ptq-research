"""MixFP4 Algorithm 1 测试。"""

from __future__ import annotations

import torch

from ptq_fp4_reference.formats import FORMAT_SPECS, positive_levels
from ptq_fp4_reference.mixfp4 import mixfp4_fake_quant


def test_e1m2_levels_match_paper_definition() -> None:
    levels = positive_levels(
        FORMAT_SPECS["e1m2"], device=torch.device("cpu"), dtype=torch.float32
    )
    torch.testing.assert_close(levels, torch.arange(0.0, 4.0, 0.5))


def test_mixfp4_selects_the_lower_error_candidate_per_block() -> None:
    values = torch.tensor(
        [[float(i) / 7 for i in range(-8, 8)] + [0.0, 0.2, 0.4, 0.6] * 4]
    )
    result = mixfp4_fake_quant(values)
    selected_mse = torch.where(result.use_e1m2, result.e1m2_mse, result.e2m1_mse)
    best_mse = torch.minimum(result.e1m2_mse, result.e2m1_mse)
    torch.testing.assert_close(selected_mse, best_mse)
    assert result.quantized_values.shape == values.shape
    assert result.use_e1m2.shape == (1, 2)


def test_mixfp4_handles_zero_tensor_and_padding() -> None:
    values = torch.zeros(3, 19, dtype=torch.bfloat16)
    result = mixfp4_fake_quant(values)
    torch.testing.assert_close(result.quantized_values, values)
    assert result.quantized_values.dtype == torch.bfloat16
    assert torch.isfinite(result.block_scales).all()
