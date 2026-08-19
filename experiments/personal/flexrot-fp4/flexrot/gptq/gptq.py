from __future__ import annotations

from dataclasses import dataclass

import torch

from .mse_grid import PreparedFormatProjector
from .static_actorder import static_actorder


@dataclass(frozen=True)
class GPTQFactor:
    """阻尼 Hessian 的逆 Cholesky 因子及静态遍历置换。"""

    inverse_cholesky: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    damp: float
    static_actorder_enabled: bool
    dead_columns: int


@dataclass(frozen=True)
class GPTQResult:
    quantized_weight: torch.Tensor
    trace: tuple[dict[str, float | int], ...]


@torch.no_grad()
def build_hessian(activations: torch.Tensor) -> torch.Tensor:
    """由校准激活构造 ``H = 2 X.T X / n`` 的 Gram/Hessian 近似。

    GPTQ 的二次误差模型来自线性层输出误差；对固定输入 X，其曲率由输入
    通道的 Gram 矩阵决定。这里只保存当前层/当前共享输入组的 H，完成该层
    后立即释放，避免 7B/8B 在单 4090 上累计全部 Hessian。
    """

    if activations.ndim != 2:
        activations = activations.reshape(-1, activations.shape[-1])
    values = activations.float()
    if values.shape[0] == 0:
        raise ValueError("calibration activations 不能为空")
    return values.t().matmul(values).mul_(2.0 / values.shape[0])


@torch.no_grad()
def prepare_gptq_factor(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    use_static_actorder: bool = False,
    percdamp: float = 0.01,
) -> GPTQFactor:
    """对 Hessian 加阻尼并构造 GPTQ 误差补偿所需的逆 Cholesky 因子。"""

    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("weight/Hessian 形状不一致")
    width = weight.shape[1]
    if use_static_actorder:
        permutation, inverse_permutation = static_actorder(hessian)
    else:
        permutation = torch.arange(width, device=weight.device)
        inverse_permutation = permutation
    h = hessian.float().index_select(0, permutation).index_select(1, permutation).clone()
    reordered_weight = weight.float().index_select(1, permutation)
    dead = reordered_weight.eq(0).all(dim=0)
    if dead.any():
        h[dead, :] = 0
        h[:, dead] = 0
        h[dead, dead] = 1
    # damping 抑制病态或近奇异 Gram 方向，使 Cholesky 在低样本校准下稳定。
    damp = float(percdamp * torch.diag(h).mean().item())
    diagonal = torch.arange(width, device=h.device)
    h[diagonal, diagonal] += damp
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
    inverse_cholesky = torch.linalg.cholesky(inverse, upper=True)
    return GPTQFactor(
        inverse_cholesky=inverse_cholesky,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        damp=damp,
        static_actorder_enabled=use_static_actorder,
        dead_columns=int(dead.sum().item()),
    )


@torch.no_grad()
def gptq_quantize(
    weight: torch.Tensor,
    projector: PreparedFormatProjector,
    factor: GPTQFactor,
    *,
    compensate: bool = True,
    traversal_chunk: int = 128,
) -> GPTQResult:
    """逐列量化权重，并用逆 Hessian 结构把当前列误差补偿到未量化列。

    W4A4 中 GPTQ 只优化 weight：权重是静态参数，能够逐列更新；activation
    取决于每次输入，运行时按校准好的 NV global scale 或 MX block scale 做
    fake quant。把 activation 也塞进 GPTQ 权重更新会混淆两种误差来源。
    """

    width = weight.shape[1]
    if factor.permutation.numel() != width:
        raise ValueError("factor 与 weight 宽度不匹配")
    traversal = factor.permutation
    working = weight.float().index_select(1, traversal).clone()
    quantized = torch.empty_like(working)
    hinv = factor.inverse_cholesky
    traces: list[dict[str, float | int]] = []

    # traversal_chunk 只控制误差传播的计算分块，不改变权重 physical layout。
    for chunk_index, start in enumerate(range(0, width, traversal_chunk)):
        end = min(start + traversal_chunk, width)
        block = working[:, start:end].clone()
        errors = torch.zeros_like(block)
        local_hinv = hinv[start:end, start:end]
        chunk_error = 0.0
        for local_column in range(end - start):
            traversal_column = start + local_column
            physical_column = int(traversal[traversal_column].item())
            column = block[:, local_column]
            qcolumn = projector.project_column(column, physical_column)
            quantized[:, traversal_column] = qcolumn
            column_error = column - qcolumn
            chunk_error += float(column_error.double().square().sum().item())
            if compensate:
                normalized_error = column_error / local_hinv[local_column, local_column]
                block[:, local_column:].sub_(
                    normalized_error.unsqueeze(1)
                    * local_hinv[local_column, local_column:]
                )
                errors[:, local_column] = normalized_error
        if compensate and end < width:
            working[:, end:].sub_(errors.matmul(hinv[start:end, end:]))
        traces.append(
            {
                "chunk_index": chunk_index,
                "completed_columns": end,
                "column_error_norm": chunk_error**0.5,
            }
        )

    # 恢复物理列顺序，保证后续 Linear 仍接收原模型的通道排列。
    physical = quantized.index_select(1, factor.inverse_permutation).contiguous()
    return GPTQResult(
        quantized_weight=physical.to(weight.dtype), trace=tuple(traces)
    )
