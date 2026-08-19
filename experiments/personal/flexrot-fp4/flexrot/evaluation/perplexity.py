from __future__ import annotations

import math
import time

import torch
import torch.nn as nn


@torch.inference_mode()
def evaluate_perplexity(
    model: nn.Module,
    windows: list[torch.Tensor],
    *,
    device: torch.device | str,
) -> tuple[dict, list[dict]]:
    """计算连续 WikiText-2 窗口的 causal LM PPL。

    每个长度 512 的窗口预测 511 个 token；8 个窗口必须得到 4088 个预测
    token。显式记录该数量可防止数据切分或标签位移发生静默变化。
    """

    loss = nn.CrossEntropyLoss(reduction="sum")
    rows: list[dict] = []
    total_nll, total_tokens = 0.0, 0
    started = time.perf_counter()
    for index, cpu_ids in enumerate(windows):
        ids = cpu_ids.to(device)
        logits = model(input_ids=ids, use_cache=False).logits
        labels = ids[:, 1:]
        nll = float(
            loss(logits[:, :-1].reshape(-1, logits.shape[-1]), labels.reshape(-1)).item()
        )
        tokens = int(labels.numel())
        rows.append({"window": index, "total_nll": nll, "mean_nll": nll / tokens, "token_count": tokens})
        total_nll += nll
        total_tokens += tokens
        del ids, logits, labels
    mean_nll = total_nll / total_tokens
    return {
        "ppl": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "total_nll": total_nll,
        "predicted_tokens": total_tokens,
        "evaluation_seconds": time.perf_counter() - started,
    }, rows
