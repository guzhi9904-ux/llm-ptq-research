"""MixFP4 Algorithm 1 的 PyTorch fake-quant 版本。

实现逐个 16 元素 block 比较 E2M1 和 E1M2 的反量化 MSE，并把格式选择保存在
布尔 mask 中。这里只检查论文算法，不包含论文提出的 E2M2 Tensor Core 改动。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .formats import FORMAT_SPECS, _reshape_blocks, _restore_blocks, nearest_float_value


MIXFP4_BLOCK_SIZE = 16
MIXFP4_GLOBAL_DIVISOR = 2688.0


@dataclass
class MixFP4Result:
    """保存反量化结果以及每个 block 的格式和 scale。"""

    quantized_values: Tensor
    use_e1m2: Tensor
    block_scales: Tensor
    tensor_scale: Tensor
    e2m1_mse: Tensor
    e1m2_mse: Tensor


def _encode_unsigned_e4m3(values: Tensor) -> Tensor:
    encoded = nearest_float_value(values, FORMAT_SPECS["e4m3"])
    return encoded.clamp_min(torch.finfo(values.dtype).tiny)


def _quantize_e1m2_as_int4(values: Tensor) -> Tensor:
    """按论文式 38，把 E1M2 的 0.5 间隔乘 2 映射到 0..7。"""

    return 2.0 * nearest_float_value(values / 2.0, FORMAT_SPECS["e1m2"])


def mixfp4_fake_quant(values: Tensor) -> MixFP4Result:
    """按 MixFP4 Algorithm 1 执行逐 block RTN 和静态格式选择。"""

    if not values.is_floating_point():
        raise TypeError("MixFP4 输入必须是浮点张量")
    original_dtype = values.dtype
    working = values.float() if values.dtype in {torch.float16, torch.bfloat16} else values
    blocks, padding = _reshape_blocks(working, MIXFP4_BLOCK_SIZE)

    tensor_absmax = working.abs().amax()
    tensor_scale = tensor_absmax / MIXFP4_GLOBAL_DIVISOR
    tensor_scale = torch.where(
        tensor_absmax == 0,
        torch.ones_like(tensor_scale),
        tensor_scale,
    )
    normalized = blocks / tensor_scale
    block_absmax = normalized.abs().amax(dim=-1, keepdim=True)

    # Algorithm 1：E2M1 最大幅值为 6；E1M2 乘 2 后最大幅值为 7。
    e2_scale = _encode_unsigned_e4m3(block_absmax / 6.0)
    e1_scale = _encode_unsigned_e4m3(block_absmax / 7.0)

    e2_payload = nearest_float_value(normalized / e2_scale, FORMAT_SPECS["e2m1"])
    e1_payload = _quantize_e1m2_as_int4(normalized / e1_scale)
    e2_reconstructed = e2_payload * e2_scale
    e1_reconstructed = e1_payload * e1_scale
    e2_mse = (e2_reconstructed - normalized).square().mean(dim=-1)
    e1_mse = (e1_reconstructed - normalized).square().mean(dim=-1)

    # 论文伪代码在 Err_E2M1 < Err_E1M2 时选 E2M1，因此相等时落到 E1M2。
    use_e1m2 = e1_mse <= e2_mse
    selected = torch.where(use_e1m2.unsqueeze(-1), e1_reconstructed, e2_reconstructed)
    selected_scale = torch.where(use_e1m2.unsqueeze(-1), e1_scale, e2_scale)
    restored = _restore_blocks(selected * tensor_scale, padding).to(original_dtype)
    return MixFP4Result(
        quantized_values=restored,
        use_e1m2=use_e1m2,
        block_scales=selected_scale.squeeze(-1),
        tensor_scale=tensor_scale,
        e2m1_mse=e2_mse,
        e1m2_mse=e1_mse,
    )
