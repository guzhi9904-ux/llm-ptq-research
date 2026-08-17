# GPTQ

## 方法定位

GPTQ 是本仓库的二阶 weight-only PTQ 起点，对应 ICLR 2023 论文《GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers》。官方代码为 [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)。

当前状态：`source-pinned`。官方 `main@2d65066eeb06a5c9ff5184d8cebdf33662c67faf` 已导入 `upstream/`。不能用 AutoGPTQ、QuaRot GPTQ 或 FP-Quant GPTQ 代替原论文实现。

## 核心逻辑

1. 用校准激活近似当前线性层的 Hessian/Gram 矩阵。
2. 按列逐步量化权重；每次量化后计算当前列误差。
3. 使用逆 Hessian 信息把误差补偿到尚未量化的列。
4. 通过 block update 降低完整逐列更新的计算开销。
5. 可选 `ActOrder` 按 activation/Hessian 对角线的重要性改变遍历顺序；可选 `true-sequential` 在 Transformer block 内继续按真实执行顺序校准。

GPTQ 的关键不是“选择一个更好的 scale”，而是利用输入相关的二阶几何重新分配量化误差。

## 核心实验

固定源码提供以下三组核心实验：

- RTN 与 GPTQ 的 W4A16 对照。
- GPTQ 开关 `ActOrder` 的消融。
- GPTQ 开关 `true-sequential` 的消融。

LLaMA W4A16 的官方示例为：

```bash
# 使用 C4 校准；开启 block 内 true-sequential 和 ActOrder，并采用新版评测预处理。
python llama.py /LLaMA_HF_模型路径 c4 \
  --wbits 4 \
  --true-sequential \
  --act-order \
  --new-eval
```

OPT-125M 的 RTN/GPTQ 对照为：

```bash
# RTN：--nearest 表示只做最近邻舍入，不做二阶补偿。
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --nearest

# GPTQ：不带 --nearest；groupsize 可按论文实验显式设置。
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4
```

## 必须核实的参数

| 参数 | 中文说明 | 当前状态 |
|---|---|---|
| `wbits` | 权重量化位宽 | 官方支持 2/3/4 bit |
| `groupsize` | 沿输入通道共享 scale/zero-point 的权重组大小 | 示例可设 1024；LLaMA 表格包含 g128 |
| `percdamp` | 相对 damping；用于稳定 Hessian 求逆 | 具体默认值以 `gptq.py`/各入口 argparse 为准 |
| `blocksize` | GPTQ 误差更新的计算 block，不等于 quantization group | 具体默认值以固定代码为准 |
| `act-order` | 按 activation/Hessian 重要性改变列顺序 | LLaMA 示例开启 |
| `true-sequential` | 在 Transformer block 内按真实子层顺序量化 | LLaMA 示例开启 |
| `static-groups` | 预先固定 quantization group 网格 | 可避免 ActOrder 造成运行时 group 重排 |
| `new-eval` | 使用 camera-ready 更新后的 C4/PTB 预处理 | 新结果应显式开启 |

## 复现边界

- 原始 GPTQ 主要是 weight-only；W4A4 中的 GPTQ 通常来自后续仓库扩展。
- 不同仓库的 damping、网格、group 物理布局和 ActOrder 实现可能不同，不能只看方法名合并结果。
