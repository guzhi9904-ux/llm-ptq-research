from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn


class ActivationCollector:
    """以 forward-pre-hook 收集 Linear 输入，并立即移到 CPU。

    这样 GPU 仅保留当前 forward 的激活；缓存不会与 BF16 模型、当前 Hessian
    同时占用 4090 显存。正式大模型流程应在每层完成后把张量写盘并清空本对象。
    """

    def __init__(self, module_names: set[str] | None = None) -> None:
        self.module_names = module_names
        self.values: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("collector 已安装")
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.module_names is not None and name not in self.module_names:
                continue

            def hook(_module, args, *, key=name):
                self.values[key].append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())

            self._handles.append(module.register_forward_pre_hook(hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def merged(self, name: str) -> torch.Tensor:
        if not self.values[name]:
            raise KeyError(name)
        return torch.cat(self.values[name], dim=0)

    def clear(self) -> None:
        self.values.clear()
