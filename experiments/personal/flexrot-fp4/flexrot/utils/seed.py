from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定 Python/NumPy/PyTorch 随机性；正式结果仍需记录软件和 GPU 版本。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
