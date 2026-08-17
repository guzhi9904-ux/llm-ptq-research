# MR-GPTQ 与 FP-Quant

## 方法定位

MR-GPTQ 出自 ICLR 2026 论文《Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization》。官方代码仓库名为 [IST-DASLab/FP-Quant](https://github.com/IST-DASLab/FP-Quant)。

本地 commit 为 `d2e3092f968262c4de5fb050e1aef568a280dadd`，但有 5 项未提交变化，并且根目录未发现 LICENSE；因此当前只做逻辑参考，不直接复制源码。

## 核心逻辑

1. 用格式正确的 MXFP4/NVFP4 quantizer 建立 RTN/GPTQ 基线。
2. 对 weight 使用可离线融合的 block-wise Hadamard transform。
3. 对 activation 在运行时做对应的 Hadamard transform，保持 Linear 输出等价。
4. 在旋转坐标系内执行 GPTQ，把 rotation 造成的 weight quantization damage 通过二阶补偿转移到较不敏感方向。
5. 按格式选择 scale 策略：MXFP4 重点处理 E8M0 scale 误差，NVFP4 可使用 MSE-optimized scale。

## 核心实验：官方脚本变量

```bash
# W4A4 NVFP4 + GPTQ + Hadamard；校准使用 128 条 FineWeb-Edu、长度 2048。
MODEL=meta-llama/Llama-3.2-1B-Instruct \
FORMAT=nvfp \
W_BITS=4 \
A_BITS=4 \
W_GROUP_SIZE=16 \
A_GROUP_SIZE=16 \
GPTQ=1 \
TRANSFORM_CLASS=hadamard \
HADAMARD_GROUP_SIZE=128 \
NUM_SEQUENCES=128 \
bash /固定后的官方运行脚本.sh
```

官方 README 最终调用：

```bash
python model_quant.py \
  --model_name_or_path /模型路径 \
  --format nvfp \
  --w_bits 4 \
  --a_bits 4 \
  --w_group_size 16 \
  --a_group_size 16 \
  --transform_class hadamard \
  --gptq \
  --hadamard_group_size 128 \
  --dataset_name_or_path fineweb-edu \
  --num_sequences 128 \
  --sequence_length 2048 \
  --eval_perplexity
```

## 参数说明

| 参数 | 官方 README 默认/示例 | 中文说明 |
|---|---:|---|
| `format` | `nvfp` | 选择 NVFP4；MXFP4 对应值需从 argparse 核实 |
| `w_bits/a_bits` | 4/16 默认 | W4A4 必须显式把 `a_bits` 改成 4 |
| `w_group_size/a_group_size` | 16/16 | 默认与 NVFP4 原生 block 对齐 |
| `gptq` | 默认关闭 | 开启后才是 GPTQ/MR-GPTQ，不开启是 RTN |
| `transform_class` | `identity` 默认 | MR 路径需显式选择 Hadamard 类变换 |
| `hadamard_group_size` | 128 | 论文/仓库默认 transform group，不等于 MXFP4 32 或 NVFP4 16 的原生 block |
| `num_sequences` | 128 | 校准序列数 |
| `sequence_length` | 2048 | 校准长度 |
| `w_observer` | `minmax` 默认 | 可切换 MSE scale search；两者结果必须分开 |
| `quantization_order` | `default` | Static ActOrder 需要显式配置 |

## 复现边界

- 官方入口当前只支持 LLaMA 与 Qwen3；Qwen2.5 需要 adapter。
- `realquant` 导出与 pseudo quant 精度评测分开记录。
- Hadamard group 128 是仓库/论文选择，不是 microscaling 原生 group。

## 本仓库独立参考实现

由于官方固定版本没有发现覆盖整个仓库的 LICENSE，官方源码继续只保留在本地。公开仓库提供了根据论文公式重新编写的 [`fp4/reference/`](../../reference/)：包含 FP4 网格、block Hadamard、MSE scale search、static ActOrder 和 GPTQ error compensation，不包含 QuTLASS kernel，也没有复制 FP-Quant 源文件。
