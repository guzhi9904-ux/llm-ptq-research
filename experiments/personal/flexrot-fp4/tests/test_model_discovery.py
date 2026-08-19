"""用微型模型检查七类受支持模块的发现规则，不依赖下载真实大模型。"""

import torch.nn as nn

from flexrot.models.adapters import discover_linear_modules


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.down_proj = nn.Linear(16, 16)
        self.unrelated = nn.Linear(16, 16)


def test_discovers_only_supported_linear_names() -> None:
    found = discover_linear_modules(TinyBlock())
    assert set(found) == {"q_proj", "down_proj"}
