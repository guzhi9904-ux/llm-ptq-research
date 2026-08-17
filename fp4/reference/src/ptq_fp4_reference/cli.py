"""对已提取 Linear weight 和校准 activation 执行参考量化。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .micromix import build_micromix_plan, micromix_fake_quant
from .mr_gptq import MRGPTQConfig, quantize_mr_gptq


def load_tensor(path: Path) -> Tensor:
    """只接受 tensor 或仅含 tensor 字段的安全 PyTorch 文件。"""

    value: Any = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, Tensor):
        return value
    if isinstance(value, dict) and isinstance(value.get("tensor"), Tensor):
        return value["tensor"]
    raise ValueError(f"文件必须直接保存 Tensor，或包含 tensor 字段：{path}")


def run_mr_gptq(args: argparse.Namespace, weight: Tensor, activations: Tensor) -> dict[str, Any]:
    config = MRGPTQConfig(
        format_name=args.format,
        use_rotation=not args.no_rotation,
        hadamard_group_size=args.hadamard_group_size,
        use_gptq=not args.rtn,
        static_act_order=not args.default_order,
        scale_strategy=args.scale_strategy,
        fit_mxfp_range=args.fit_mxfp_range,
        relative_damp=args.relative_damp,
        update_block_size=args.update_block_size,
        mse_search_steps=args.mse_search_steps,
    )
    result = quantize_mr_gptq(weight, activations, config)
    return {
        "method": "mr-gptq-reference",
        "config": vars(config),
        "quantized_weight": result.quantized_weight.cpu(),
        "column_order": result.column_order.cpu(),
        "group_scales": result.scale_plan.scales.cpu(),
        "weight_mse": result.weight_mse,
    }


def run_micromix(args: argparse.Namespace, weight: Tensor, activations: Tensor) -> dict[str, Any]:
    plan = build_micromix_plan(activations, channel_alignment=args.channel_alignment)
    permutation = plan.permutation
    reordered_weight = weight[:, permutation]
    quantized_weight = micromix_fake_quant(
        reordered_weight, plan, fp8_variant=args.fp8_variant
    )
    return {
        "method": "micromix-reference",
        "config": {
            "channel_alignment": args.channel_alignment,
            "fp8_variant": args.fp8_variant,
        },
        "quantized_weight": quantized_weight.cpu(),
        "permutation": permutation.cpu(),
        "channel_scores": plan.channel_scores.cpu(),
        "partitions": [vars(item) for item in plan.partitions],
        "threshold_proportions": plan.threshold_proportions,
        "average_element_bits": plan.average_element_bits,
        "average_storage_bits": plan.average_storage_bits,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MR-GPTQ/MicroMix 独立张量参考实现")
    parser.add_argument("method", choices=["mr-gptq", "micromix"])
    parser.add_argument("--weight", type=Path, required=True, help="[out,in] weight Tensor")
    parser.add_argument(
        "--activations", type=Path, required=True, help="[...,in] 校准 activation Tensor"
    )
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--format", choices=["mxfp4", "nvfp4"], default="mxfp4")
    parser.add_argument("--no-rotation", action="store_true")
    parser.add_argument("--hadamard-group-size", type=int, default=128)
    parser.add_argument("--rtn", action="store_true", help="关闭 GPTQ，生成 RTN 基线")
    parser.add_argument("--default-order", action="store_true", help="关闭 static ActOrder")
    parser.add_argument("--scale-strategy", choices=["minmax", "mse"], default="mse")
    parser.add_argument("--fit-mxfp-range", action="store_true")
    parser.add_argument("--relative-damp", type=float, default=0.01)
    parser.add_argument("--update-block-size", type=int, default=128)
    parser.add_argument("--mse-search-steps", type=int, default=33)

    parser.add_argument("--channel-alignment", type=int, default=1)
    parser.add_argument("--fp8-variant", choices=["e5m2", "e4m3"], default="e5m2")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    weight = load_tensor(args.weight).float()
    activations = load_tensor(args.activations).float()
    if weight.ndim != 2 or weight.shape[-1] != activations.shape[-1]:
        raise ValueError("weight 必须为 [out,in]，且 activation 最后一维必须等于 in")
    payload = (
        run_mr_gptq(args, weight, activations)
        if args.method == "mr-gptq"
        else run_micromix(args, weight, activations)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        key: value
        for key, value in payload.items()
        if not isinstance(value, Tensor) and key != "config"
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
