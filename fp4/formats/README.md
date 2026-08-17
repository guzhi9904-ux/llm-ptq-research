# FP4 格式定义

## 基础 E2M1

本仓库用 `E2M1` 表示 1 个 sign bit、2 个 exponent bit、1 个 mantissa bit 的 FP4 元素格式。有效有限幅值与零值的精确定义必须以对应标准和官方量化器测试为准，不能只按 `torch.float8` 类比实现。

## MXFP4

- 数据 block：通常为 32 个 FP4 元素。
- 元素格式：E2M1。
- block scale：E8M0，等价于 power-of-two scale。
- 平均存储预算通常按 4 bit 元素加共享 scale 计算。
- scale rounding 是主要误差来源之一，必须明确是向上、nearest 还是搜索得到。

## NVFP4

- 数据 block：通常为 16 个 FP4 元素。
- 元素格式：E2M1。
- local block scale：E4M3。
- 另有 per-tensor/global scale，形成两级 scale hierarchy。
- 不能只实现 block E4M3 scale 而忽略 global scale。

## 必做测试

- 所有正负 code、零值、最大幅值和饱和边界。
- scale 的编码、舍入和解码。
- block 跨界和尾块 padding。
- weight 与 activation 的布局方向。
- fake quant 与真实 pack/unpack/kernel 的一致性。
