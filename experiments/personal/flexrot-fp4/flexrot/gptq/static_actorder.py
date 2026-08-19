from __future__ import annotations

import torch


def static_actorder(hessian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Hessian 对角线从大到小产生静态列遍历顺序及其逆置换。

    Hessian 对角线近似每个输入通道的激活能量；高能量列先量化，便于后续
    列吸收误差。这里改变的是 GPTQ 的 traversal order，不改变权重的
    physical layout：scale group 仍按原物理列号查找。量化结束必须用逆置换
    恢复原列顺序，否则线性层会把通道接错。
    """

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian 必须为方阵")
    permutation = torch.argsort(torch.diag(hessian), descending=True)
    return permutation, torch.argsort(permutation)
