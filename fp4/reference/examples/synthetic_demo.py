#!/usr/bin/env python3
"""用合成张量运行 MR-GPTQ 和 MicroMix，不下载模型或数据集。"""

from __future__ import annotations

import argparse

import torch

from ptq_fp4_reference.micromix import build_micromix_plan, micromix_fake_linear
from ptq_fp4_reference.mr_gptq import (
    MRGPTQConfig,
    mr_gptq_fake_linear,
    quantize_mr_gptq,
)


def run_mr_gptq(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    weight = torch.randn(args.output_size, args.hidden_size)
    calibration = torch.randn(args.samples, args.hidden_size)
    config = MRGPTQConfig(
        hadamard_group_size=args.hadamard_group_size,
        mse_search_steps=9,
        update_block_size=32,
    )
    result = quantize_mr_gptq(weight, calibration, config)
    output = mr_gptq_fake_linear(calibration[:8], result, config)
    print(f"MR-GPTQ weight MSE：{result.weight_mse:.8f}")
    print(f"输出形状：{tuple(output.shape)}")


def run_micromix(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    weight = torch.randn(args.output_size, args.hidden_size)
    calibration = torch.randn(args.samples, args.hidden_size) * 0.01
    calibration[:, -8:] *= 500.0
    plan = build_micromix_plan(
        calibration, channel_alignment=args.channel_alignment
    )
    output = micromix_fake_linear(calibration[:8], weight, plan)
    counts = {item.bits: item.channels for item in plan.partitions}
    print(f"channel 分配：{counts}")
    print(f"平均元素位宽：{plan.average_element_bits:.3f}")
    print(f"含 E8M0 scale 的平均存储位宽：{plan.average_storage_bits:.3f}")
    print(f"输出形状：{tuple(output.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FP4 PTQ 合成张量示例")
    parser.add_argument("method", choices=["mr-gptq", "micromix"])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--hadamard-group-size", type=int, default=128)
    parser.add_argument("--channel-alignment", type=int, default=8)
    args = parser.parse_args()
    run_mr_gptq(args) if args.method == "mr-gptq" else run_micromix(args)


if __name__ == "__main__":
    main()
