from __future__ import annotations

import random
from pathlib import Path

import torch
from datasets import load_dataset


def load_token_windows(
    tokenizer,
    *,
    dataset_path: str | Path | None = None,
    split: str,
    seq_len: int,
    num_windows: int | None,
    dataset_id: str = "Salesforce/wikitext",
    subset: str = "wikitext-2-raw-v1",
    cache_dir: str | Path | None = None,
    starts: list[int] | None = None,
    filter_blank: bool = True,
) -> list[torch.Tensor]:
    """按连续 token 流切 WikiText-2 窗口，禁止跨 split 混用校准与评测。

    ``dataset_path`` 指向 datasets 保存到磁盘的目录时优先离线读取；否则由
    Hugging Face datasets 下载。每个窗口形状为 [1, seq_len]。
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("split 必须为 train/validation/test")
    if dataset_path is not None and Path(dataset_path).exists():
        from datasets import load_from_disk

        stored = load_from_disk(str(dataset_path))
        dataset = stored[split] if hasattr(stored, "keys") else stored
    else:
        dataset = load_dataset(
            dataset_id, subset, split=split, cache_dir=None if cache_dir is None else str(cache_dir)
        )
    rows = [str(item) for item in dataset["text"]]
    if filter_blank:
        rows = [row for row in rows if row.strip()]
    text = "\n\n".join(rows)
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.reshape(-1)
    resolved_count = token_ids.numel() // seq_len if num_windows is None or num_windows <= 0 else num_windows
    resolved_starts = (
        [index * seq_len for index in range(resolved_count)]
        if starts is None
        else [int(value) for value in starts]
    )
    if len(resolved_starts) != resolved_count:
        raise ValueError("starts 数量必须等于 num_windows")
    if max(resolved_starts, default=0) + seq_len > token_ids.numel():
        raise ValueError("token stream 不足以覆盖请求的窗口起点")
    return [token_ids[start : start + seq_len].unsqueeze(0).contiguous() for start in resolved_starts]


def load_fineweb_edu_windows(
    tokenizer,
    *,
    seq_len: int,
    num_sequences: int,
    seed: int,
    cache_dir: str | Path | None = None,
    shuffle_buffer_size: int = 1_000,
) -> list[torch.Tensor]:
    """按官方 FP-Quant 采样语义流式读取 FineWeb-Edu 校准序列。

    使用 ``sample-10BT``、streaming shuffle 和每篇文档内的随机连续窗口；
    独立 ``Random`` 实例保证不同 seed 可复现且不污染进程级随机状态。
    数据只流式消费到收满请求序列，不会下载完整 10B-token 子集。
    """

    if seq_len <= 0 or num_sequences <= 0:
        raise ValueError("seq_len 和 num_sequences 必须为正数")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
        cache_dir=None if cache_dir is None else str(cache_dir),
    ).shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    generator = random.Random(seed)
    windows: list[torch.Tensor] = []
    for sample in dataset:
        encoded = tokenizer(
            str(sample["text"]),
            return_tensors="pt",
        ).input_ids
        if encoded.shape[1] < seq_len:
            continue
        start = generator.randint(0, encoded.shape[1] - seq_len)
        windows.append(encoded[:, start : start + seq_len].contiguous())
        if len(windows) == num_sequences:
            return windows
    raise RuntimeError(
        f"FineWeb-Edu 流结束前仅得到 {len(windows)}/{num_sequences} 个校准窗口"
    )
