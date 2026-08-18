# MR-GPTQ 与 FP-Quant

## 方法定位

MR-GPTQ 出自 ICLR 2026 论文《Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization》。官方代码仓库名为 [IST-DASLab/FP-Quant](https://github.com/IST-DASLab/FP-Quant)。

`upstream/` 是从官方 commit `d2e3092f968262c4de5fb050e1aef568a280dadd` 直接导出的干净快照，没有带入本地 5 项修改。根目录没有 LICENSE；本仓库按用户确认的上传授权保留源码，同时把这一点写在 `SOURCE.md` 中。

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

## 本仓库里的复现代码

官方源码已经放在 [`upstream/`](upstream/)；完整模型入口是 `model_quant.py`，量化主体在 `src/quantization/`，变换代码在 `src/transforms/`。另有一份容易读和测试的独立版本放在 [`fp4/reference/`](../../reference/)，包括 FP4 网格、block Hadamard、MSE scale search、static ActOrder 和 GPTQ error compensation。

论文正文的主实验使用 1024 条 FineWeb 校准序列；官方 README 的快速命令默认示例使用 128 条 FineWeb-Edu。复现实验需要明确写清使用哪一套，不能把两者当成同一个设置。
