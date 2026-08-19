"""检查 FlexRot 正交性、45° native Hadamard 端点和线性层成对变换等价性。"""

import torch

from flexrot.rotation.hadamard import normalized_hadamard
from flexrot.rotation.transforms import build_rotation, rotate_linear_problem


def test_rotation_is_orthogonal() -> None:
    for block_size in (16, 32):
        for theta in (0.0, 5.625, 22.5, 45.0):
            rotation = build_rotation(block_size, theta)
            identity = torch.eye(block_size, dtype=rotation.dtype)
            torch.testing.assert_close(rotation.T @ rotation, identity, atol=1e-12, rtol=1e-12)


def test_45_degrees_equals_native_hadamard() -> None:
    for block_size in (16, 32):
        torch.testing.assert_close(
            build_rotation(block_size, 45.0),
            normalized_hadamard(block_size),
            atol=0,
            rtol=0,
        )


def test_transform_only_linear_output_is_equivalent() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(5, 64, generator=generator, dtype=torch.float64)
    weight = torch.randn(11, 64, generator=generator, dtype=torch.float64)
    xr, wr = rotate_linear_problem(x, weight, block_size=16, theta_deg=5.625)
    torch.testing.assert_close(xr @ wr.T, x @ weight.T, atol=1e-10, rtol=1e-10)
