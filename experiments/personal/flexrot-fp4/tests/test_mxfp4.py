"""检查 MXFP4 block32、E8M0 二次幂 scale 与全零块部署语义。"""

import torch

from flexrot.quantization.mxfp4 import fake_quant_mxfp4


def test_legacy_mxfp4_zero_block_uses_half_scale_but_exact_zero_output() -> None:
    values = torch.zeros(3, 32)
    result = fake_quant_mxfp4(values, scale_mode="legacy")
    torch.testing.assert_close(result.scales, torch.full((3, 1), 0.5))
    assert result.zero_block_mask.all()
    assert torch.equal(result.dequantized, values)


def test_mxfp4_scales_are_powers_of_two() -> None:
    values = torch.arange(1, 65, dtype=torch.float32).reshape(2, 32)
    scales = fake_quant_mxfp4(values, scale_mode="legacy").scales
    assert torch.equal(torch.log2(scales), torch.log2(scales).round())


def test_paper_mxfp4_scale_fitting_matches_official_formula() -> None:
    values = torch.full((1, 32), 5.0)
    result = fake_quant_mxfp4(values, scale_mode="paper")
    # floor(log2(5)) -> 2；官方有效 scale = 2**2 / 3。
    torch.testing.assert_close(result.scales, torch.tensor([[4.0 / 3.0]]))
    assert torch.equal(result.dequantized, torch.full_like(values, 16.0 / 3.0))


def test_paper_mxfp4_zero_block_uses_official_underflow_scale() -> None:
    result = fake_quant_mxfp4(torch.zeros(1, 32), scale_mode="paper")
    assert result.scale_codes.item() == -128
    assert result.scales.item() > 0.0
    assert result.zero_block_mask.all()
    assert not result.dequantized.any()
