from __future__ import annotations

from dataclasses import dataclass

import torch

from .nvfp4 import E2M1_MAX, E4M3_MAX


@dataclass
class NVFP4GlobalScaleObserver:
    """流式记录 activation amax，避免把全部校准激活同时放进 GPU。"""

    maximum: float = 0.0
    batches: int = 0

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        self.maximum = max(self.maximum, float(values.detach().abs().max().item()))
        self.batches += 1

    def compute(self) -> float:
        if self.batches == 0:
            raise RuntimeError("observer 尚未收到校准数据")
        return self.maximum / (E2M1_MAX * E4M3_MAX) if self.maximum else 1.0
