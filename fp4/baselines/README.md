# FP4 基础基线

FP4 部分固定三类基线：RTN、GPTQ 和 Rotation。每类都必须分别运行 MXFP4 与 NVFP4，不能把“FP4”当成单一格式。

## RTN

RTN 使用格式自身的 codebook、block 和 scale hierarchy 直接舍入。它是检查格式实现是否正确的第一基线。

示例配置：

```yaml
method: rtn                  # 不做 Hessian 误差补偿
format: mxfp4               # 可替换为 nvfp4，但 scale 规则必须同时变化
weight_bits: 4
activation_bits: 4
weight_block_size: 32       # MXFP4 的原生 block 大小
activation_block_size: 32
scale_format: e8m0          # MXFP4 的 power-of-two block scale
scale_selection: max_safe   # 先固定无溢出的 absmax 基线
```

## GPTQ

GPTQ 在 FP4 网格上做输入相关的逐列误差补偿。必须显式区分：

- FP4 quantization block/group。
- GPTQ 的 update block。
- ActOrder 改变的遍历顺序。
- 物理 group 是否随 ActOrder 重新排列。

## Rotation

Rotation 基线至少保留：

- identity：不旋转。
- global/full Hadamard。
- block-local Hadamard。
- MR-GPTQ 官方默认的 Hadamard group。

MXFP4 常受益于 rotation，但 NVFP4 在 absmax scale 下不一定受益。旋转强度、group size 和格式必须共同记录。

[`fp4/reference/`](../reference/) 里提供了 PyTorch 版本。运行时可以保持 FP4 网格和 scale 规则不变，只切换 identity/rotation 或 RTN/GPTQ，这样比较更直接。
