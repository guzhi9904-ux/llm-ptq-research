from __future__ import annotations

import numbers
from dataclasses import dataclass
from functools import partial
from typing import Any


@dataclass(frozen=True)
class OpenLLMTask:
    name: str
    num_fewshot: int | None = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False


# 与 MR-GPTQ 官方仓库的 OpenLLM-v1 评测参数保持一致。
PAPER_TASKS = (
    OpenLLMTask("winogrande", num_fewshot=5),
    OpenLLMTask("hellaswag", num_fewshot=10),
    OpenLLMTask("gsm8k_llama", apply_chat_template=True, fewshot_as_multiturn=True),
    OpenLLMTask("mmlu_cot_llama", apply_chat_template=True, fewshot_as_multiturn=True),
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def evaluate_paper_tasks(
    model,
    tokenizer,
    *,
    batch_size: int | str = "auto",
    seed: int = 42,
    disable_qwen3_thinking: bool = True,
    limit: int | float | None = None,
) -> dict[str, Any]:
    """在当前内存中的 fake-quant 模型上运行论文四项任务。

    不能重新从 checkpoint 加载模型，否则会丢失逐层安装的 ``FP4Linear``。
    ``lm-evaluation-harness`` 因此通过 ``HFLM(pretrained=model)`` 包装现有实例。
    """

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError(
            "缺少论文任务评测依赖；请运行 pip install -e ."
        ) from exc

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("论文的 GSM8K/MMLU-CoT 评测要求 tokenizer 提供 chat_template")
    if disable_qwen3_thinking and getattr(model.config, "model_type", None) == "qwen3":
        tokenizer.apply_chat_template = partial(
            tokenizer.apply_chat_template, enable_thinking=False
        )

    wrapped = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=4096,
    )
    manager = lm_eval.tasks.TaskManager()
    combined: dict[str, Any] = {}
    for task in PAPER_TASKS:
        payload = lm_eval.simple_evaluate(
            model=wrapped,
            tasks=[task.name],
            num_fewshot=task.num_fewshot,
            batch_size=batch_size,
            task_manager=manager,
            apply_chat_template=task.apply_chat_template,
            fewshot_as_multiturn=task.fewshot_as_multiturn,
            limit=limit,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            log_samples=False,
        )
        if payload is None:
            raise RuntimeError(f"lm-eval 未返回任务结果：{task.name}")
        combined.update(payload["results"])
    return _json_safe(combined)
