"""MicroMix 阈值、重排与混合精度 fake quant 测试。"""

from __future__ import annotations

import torch

from ptq_fp4_reference.micromix import (
    build_micromix_plan,
    estimate_precision_proportions,
    micromix_fake_linear,
    quantization_threshold,
)


def outlier_activations() -> torch.Tensor:
    torch.manual_seed(4)
    values = torch.randn(128, 96) * 0.01
    values[:, -4:] *= 1000.0
    return values


def test_mxfp6_threshold_is_larger_than_mxfp4() -> None:
    maximum = torch.tensor([10.0])
    assert quantization_threshold(maximum, 6) > quantization_threshold(maximum, 4)


def test_precision_proportions_sum_to_one() -> None:
    proportions = estimate_precision_proportions(outlier_activations())
    assert abs(sum(proportions.values()) - 1.0) < 1e-6
    assert proportions[4] > proportions[8]


def test_plan_sorts_channels_and_covers_width() -> None:
    activations = outlier_activations()
    plan = build_micromix_plan(activations, channel_alignment=8)
    sorted_scores = plan.channel_scores[plan.permutation]
    assert torch.all(sorted_scores[1:] >= sorted_scores[:-1])
    assert sum(partition.channels for partition in plan.partitions) == activations.shape[-1]
    assert 4.0 <= plan.average_element_bits <= 8.0
    assert plan.average_storage_bits == plan.average_element_bits + 0.25


def test_permuting_both_operands_preserves_full_precision_output() -> None:
    activations = outlier_activations()
    torch.manual_seed(5)
    weight = torch.randn(12, activations.shape[-1])
    plan = build_micromix_plan(activations)
    permutation = plan.permutation
    torch.testing.assert_close(
        activations[:, permutation] @ weight[:, permutation].t(),
        activations @ weight.t(),
        atol=2e-4,
        rtol=2e-4,
    )


def test_micromix_fake_linear_is_finite() -> None:
    activations = outlier_activations()
    torch.manual_seed(6)
    weight = torch.randn(12, activations.shape[-1])
    plan = build_micromix_plan(activations, channel_alignment=8)
    output = micromix_fake_linear(activations[:4], weight, plan)
    assert output.shape == (4, 12)
    assert torch.isfinite(output).all()
