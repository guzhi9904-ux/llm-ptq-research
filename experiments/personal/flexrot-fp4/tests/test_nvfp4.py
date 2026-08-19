"""检查 NVFP4 block16、E2M1、global calibration 与零块保护的基本语义。"""

import torch

from flexrot.quantization.nvfp4 import calibrate_nvfp4_global_scale, fake_quant_nvfp4


def test_nvfp4_shape_and_zero_block() -> None:
    values = torch.zeros(2, 32)
    result = fake_quant_nvfp4(values)
    assert result.dequantized.shape == values.shape
    assert result.scales.shape == (2, 2)
    assert result.zero_block_mask.all()
    assert torch.equal(result.dequantized, values)
    assert torch.isfinite(result.dequantized).all()


def test_nvfp4_calibration_and_codebook() -> None:
    values = torch.linspace(-6, 6, 32).reshape(1, -1)
    global_scale = calibrate_nvfp4_global_scale(values)
    result = fake_quant_nvfp4(values, global_scale=global_scale)
    allowed = torch.tensor([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int8)
    assert torch.isin(result.element_codes.unique(), allowed).all()
    assert result.global_scale is not None
