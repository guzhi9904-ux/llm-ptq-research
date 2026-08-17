"""block Hadamard 的正交性和 Linear 等价性测试。"""

from __future__ import annotations

import pytest
import torch

from ptq_fp4_reference.hadamard import block_hadamard, rotate_linear_pair


def test_hadamard_is_self_inverse() -> None:
    torch.manual_seed(1)
    values = torch.randn(4, 128)
    restored = block_hadamard(block_hadamard(values, 32), 32)
    torch.testing.assert_close(restored, values, atol=2e-6, rtol=2e-6)


def test_weight_activation_pair_preserves_linear_output() -> None:
    torch.manual_seed(2)
    weight = torch.randn(11, 128)
    inputs = torch.randn(17, 128)
    rotated_weight, rotated_inputs = rotate_linear_pair(weight, inputs, 128)
    torch.testing.assert_close(
        rotated_inputs @ rotated_weight.t(), inputs @ weight.t(), atol=2e-5, rtol=2e-5
    )


def test_invalid_block_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="2 的幂"):
        block_hadamard(torch.randn(2, 12), 6)
