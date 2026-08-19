"""Qwen2/Qwen3/Llama 模型注册表与 Linear 适配。"""

from .adapters import discover_linear_modules, replace_with_rtn
from .registry import load_model_spec

__all__ = ["discover_linear_modules", "load_model_spec", "replace_with_rtn"]
