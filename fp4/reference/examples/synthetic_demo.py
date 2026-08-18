#!/usr/bin/env python3
"""用合成张量运行 FP4 baseline 和论文方法，不下载模型或数据集。"""

from __future__ import annotations

import argparse

import torch

from ptq_fp4_reference.baselines import FP4BaselineConfig, quantize_fp4_baseline
from ptq_fp4_reference.micromix import build_micromix_plan, micromix_fake_linear
from ptq_fp4_reference.mixfp4 import mixfp4_fake_quant
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


def run_baseline(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    weight = torch.randn(args.output_size, args.hidden_size)
    calibration = torch.randn(args.samples, args.hidden_size)
    config = FP4BaselineConfig(
        args.baseline,
        args.format,
        hadamard_group_size=args.hadamard_group_size,
        update_block_size=32,
    )
    result = quantize_fp4_baseline(weight, calibration, config)
    print(f"基线：{args.baseline} / {args.format}")
    print(f"weight MSE：{result.weight_mse:.8f}")


def run_mixfp4(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    weight = torch.randn(args.output_size, args.hidden_size)
    result = mixfp4_fake_quant(weight)
    e1_blocks = int(result.use_e1m2.sum())
    total_blocks = result.use_e1m2.numel()
    mse = (result.quantized_values - weight).square().mean()
    print(f"E1M2 block：{e1_blocks}/{total_blocks}")
    print(f"weight MSE：{float(mse):.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FP4 PTQ 合成张量示例")
    parser.add_argument("method", choices=["baseline", "mr-gptq", "micromix", "mixfp4"])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--hadamard-group-size", type=int, default=128)
    parser.add_argument("--channel-alignment", type=int, default=8)
    parser.add_argument("--format", choices=["mxfp4", "nvfp4"], default="mxfp4")
    parser.add_argument(
        "--baseline", choices=["rtn", "rotation-rtn", "gptq"], default="rtn"
    )
    args = parser.parse_args()
    runners = {
        "baseline": run_baseline,
        "mr-gptq": run_mr_gptq,
        "micromix": run_micromix,
        "mixfp4": run_mixfp4,
    }
    runners[args.method](args)


if __name__ == "__main__":
    main()
