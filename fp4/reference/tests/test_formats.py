"""低精度浮点网格和 microscaling block 测试。"""

from __future__ import annotations

import torch

from ptq_fp4_reference.formats import (
    FORMAT_SPECS,
    MX_SPECS,
    fit_e8m0_scales,
    microscale_fake_quant,
    nearest_float_value,
    positive_levels,
)


def test_e2m1_levels_match_fp4_grid() -> None:
    levels = positive_levels(
        FORMAT_SPECS["e2m1"], device=torch.device("cpu"), dtype=torch.float32
    )
    assert levels.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_nearest_float_preserves_sign_and_saturates() -> None:
    values = torch.tensor([-9.0, -1.25, 0.0, 1.25, 9.0])
    quantized = nearest_float_value(values, FORMAT_SPECS["e2m1"])
    assert quantized.tolist() == [-6.0, -1.0, 0.0, 1.0, 6.0]


def test_microscale_padding_is_removed() -> None:
    torch.manual_seed(0)
    values = torch.randn(3, 35)
    quantized = microscale_fake_quant(values, MX_SPECS["mxfp4"])
    assert quantized.shape == values.shape
    assert torch.isfinite(quantized).all()


def test_fitted_e8m0_has_no_more_than_256_levels() -> None:
    scales = torch.pow(2.0, torch.linspace(-20, 15, 1000))
    fitted = fit_e8m0_scales(scales)
    assert torch.unique(fitted).numel() <= 256
    assert torch.all(fitted[1:] >= fitted[:-1])
