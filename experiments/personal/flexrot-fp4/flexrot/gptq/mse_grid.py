from __future__ import annotations

from dataclasses import dataclass

import torch

from flexrot.quantization.fp4_common import nearest_e2m1
from flexrot.quantization.mxfp4 import mxfp4_scales
from flexrot.quantization.nvfp4 import _e4m3_scales


@dataclass(frozen=True)
class PreparedFormatProjector:
    """GPTQ 遍历期间复用的物理分组 scale 与 FP4 列投影器。"""

    format_name: str
    scales: torch.Tensor
    raw_scales: torch.Tensor
    minmax_raw_scales: torch.Tensor
    global_scale: torch.Tensor | None
    group_size: int
    mxfp_scale_mode: str
    mse_grid: bool
    scale_search_iters: int
    max_scale_shrink_factor: float
    error_norm: float

    @property
    def groups(self) -> int:
        return int(self.scales.shape[1])

    def project_column(self, column: torch.Tensor, physical_column: int) -> torch.Tensor:
        """按原物理列所属的 scale group 投影，避免 ActOrder 打乱分组。"""

        group = physical_column // self.group_size
        scale = self.scales[:, group]
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        projected, _ = nearest_e2m1(column.float() / safe_scale)
        return projected * scale

    def project_weight(self, weight: torch.Tensor) -> torch.Tensor:
        blocks = weight.float().reshape(weight.shape[0], self.groups, self.group_size)
        safe = torch.where(self.scales > 0, self.scales, torch.ones_like(self.scales))
        projected, _ = nearest_e2m1(blocks / safe.unsqueeze(-1))
        return (projected * self.scales.unsqueeze(-1)).reshape_as(weight)


def _project_raw(
    blocks: torch.Tensor, raw_scales: torch.Tensor
) -> torch.Tensor:
    safe = torch.where(raw_scales > 0, raw_scales, torch.ones_like(raw_scales))
    projected, _ = nearest_e2m1(blocks / safe.unsqueeze(-1))
    return projected * raw_scales.unsqueeze(-1)


@torch.no_grad()
def prepare_format_projector(
    weight: torch.Tensor,
    format_name: str,
    *,
    mse_grid: bool = False,
    scale_search_iters: int = 100,
    max_scale_shrink_factor: float = 0.80,
    error_norm: float = 2.4,
    mxfp_scale_mode: str = "paper",
) -> PreparedFormatProjector:
    """为 NVFP4/MXFP4 权重选择固定物理分组 scale。

    MSE-grid 从 min-max scale 开始，搜索 ``[1-shrink, 1]`` 内的等距候选，
    目标是最小化每个块的 ``sum(abs(W-Q(W))**error_norm)``。搜索先在未编码
    raw scale 上进行，再施加 E4M3/E8M0 scale 精度。它优化量化 scale，
    rotation strength 则改变坐标系；两者是独立变量，配置中必须分别记录。
    """

    name = format_name.lower()
    if name not in {"nvfp4", "mxfp4"}:
        raise ValueError("format_name 必须为 nvfp4 或 mxfp4")
    if weight.ndim != 2:
        raise ValueError("weight 必须为 [out_features, in_features]")
    group_size = 16 if name == "nvfp4" else 32
    if weight.shape[1] % group_size:
        raise ValueError("weight 宽度不能被 FP4 group_size 整除")
    blocks = weight.float().reshape(weight.shape[0], -1, group_size)
    minmax_raw = blocks.abs().amax(dim=-1) / 6.0
    selected_raw = minmax_raw.clone()
    if mse_grid:
        best_error = torch.full_like(selected_raw, float("inf"))
        for index in range(scale_search_iters):
            shrink = 1.0 - index * max_scale_shrink_factor / scale_search_iters
            candidate = minmax_raw * shrink
            reconstructed = _project_raw(blocks, candidate)
            error = (blocks - reconstructed).abs().pow(error_norm).sum(dim=-1)
            improved = error < best_error
            best_error = torch.where(improved, error, best_error)
            selected_raw = torch.where(improved, candidate, selected_raw)

    global_scale = None
    if name == "mxfp4":
        scales, _ = mxfp4_scales(selected_raw, scale_mode=mxfp_scale_mode)
    else:
        max_raw = minmax_raw.max()
        global_scale = torch.where(max_raw > 0, max_raw / 448.0, torch.ones_like(max_raw))
        scales, _, global_scale = _e4m3_scales(selected_raw, global_scale)
    return PreparedFormatProjector(
        format_name=name,
        scales=scales,
        raw_scales=selected_raw,
        minmax_raw_scales=minmax_raw,
        global_scale=global_scale,
        group_size=group_size,
        mxfp_scale_mode=mxfp_scale_mode,
        mse_grid=mse_grid,
        scale_search_iters=scale_search_iters,
        max_scale_shrink_factor=max_scale_shrink_factor,
        error_norm=error_norm,
    )
