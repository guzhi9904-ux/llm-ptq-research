"""不依赖外部 kernel 的 block-wise normalized Hadamard transform。"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def block_hadamard(values: Tensor, block_size: int) -> Tensor:
    """在最后一维的独立 block 上应用正交 Hadamard transform。

    该实现用于数值验证；真实部署应替换成融合 kernel。最后一维必须能被
    block_size 整除，避免用 padding 悄悄改变线性层语义。
    """

    if not is_power_of_two(block_size):
        raise ValueError("Hadamard block_size 必须是 2 的幂")
    if values.shape[-1] % block_size:
        raise ValueError("最后一维必须能被 Hadamard block_size 整除")
    blocks = values.reshape(*values.shape[:-1], -1, block_size)
    result = blocks
    stride = 1
    while stride < block_size:
        paired = result.reshape(*result.shape[:-1], -1, 2, stride)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        stride *= 2
    return (result / math.sqrt(block_size)).reshape_as(values)


def rotate_linear_pair(weight: Tensor, activations: Tensor, block_size: int) -> tuple[Tensor, Tensor]:
    """同时旋转 weight 输入通道与 activation，保持全精度线性输出不变。"""

    if weight.ndim != 2 or activations.shape[-1] != weight.shape[-1]:
        raise ValueError("weight 必须为二维，且 activation 最后一维等于输入通道数")
    return block_hadamard(weight, block_size), block_hadamard(activations, block_size)
