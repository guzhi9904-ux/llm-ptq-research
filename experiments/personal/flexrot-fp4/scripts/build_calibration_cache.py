from __future__ import annotations

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import cache_path, dataset_path, model_path, repository_paths
from flexrot.calibration.cache import ActivationCache
from flexrot.calibration.data import load_token_windows
from flexrot.calibration.hooks import ActivationCollector
from flexrot.models.adapters import discover_linear_modules


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="构建逐 Linear 的 WikiText-2 校准激活缓存")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-sequences", type=int, default=8)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境不可用")
    paths = repository_paths()
    local_model = model_path(args.model, paths)
    tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True, use_fast=True)
    windows = load_token_windows(
        tokenizer,
        dataset_path=dataset_path(paths),
        split="train",
        seq_len=args.seq_len,
        num_windows=args.num_sequences,
        filter_blank=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    model.config.use_cache = False
    names = set(discover_linear_modules(model))
    collector = ActivationCollector(names)
    collector.install(model)
    for index, input_ids in enumerate(windows):
        model(input_ids=input_ids.to(args.device), use_cache=False)
        print(f"[capture] {index + 1}/{len(windows)}", flush=True)
    collector.remove()

    cache = ActivationCache(cache_path(args.model, paths))
    for index, name in enumerate(sorted(names)):
        cache.write(name, collector.merged(name))
        collector.values.pop(name, None)
        print(f"[write] {index + 1}/{len(names)} {name}", flush=True)
    cache.finalize(
        metadata={
            "model": args.model,
            "dataset": "WikiText-2 train",
            "seq_len": args.seq_len,
            "num_sequences": args.num_sequences,
            "num_tokens": args.seq_len * args.num_sequences,
            "dtype": "torch.bfloat16",
        }
    )
    del model, collector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
