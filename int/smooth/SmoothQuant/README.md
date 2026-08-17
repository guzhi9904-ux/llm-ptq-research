# SmoothQuant

## 方法定位

SmoothQuant 是 ICML 2023 的 W8A8 PTQ 方法。官方代码为 [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)，本地干净版本固定在 `c61476d728e42ae0d8a35e7e78494edcac3237b5`。

当前状态：`source-pinned`、`logic-audited`。

## 核心逻辑

对线性层输入通道 `j` 计算 activation channel max 与对应 weight channel max，并构造对角 scale：

```text
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
```

随后执行数学等价变换：activation 除以 `s_j`，weight 对应通道乘以 `s_j`。它把 activation outlier 的量化难度迁移到更容易量化的 weight，不是消除总误差。

## 关键实现

- activation scale 收集：`examples/generate_act_scales.py`
- 等价平滑：`smoothquant/smooth.py`
- fake quant：`smoothquant/fake_quant.py`
- PPL 评测：`smoothquant/ppl_eval.py`

## 核心实验一：收集校准尺度

```bash
# 官方预计算 scale 使用 Pile validation 中 512 条随机句子。
python examples/generate_act_scales.py \
  --model-name /模型路径 \
  --output-path /输出/act_scales.pt \
  --num-samples 512 \
  --seq-len 2048 \
  --dataset-path /Pile校准集路径
```

参数说明：

- `--num-samples`：校准句子数；不能与最终 PPL 评测样本重叠。
- `--seq-len`：每条校准序列长度。
- `--output-path`：保存 per-channel activation max，不是量化模型。

## 核心实验二：W8A8 PPL

```bash
# alpha 必须按模型记录；官方结果并非所有模型统一使用 0.5。
python smoothquant/ppl_eval.py \
  --model_path /模型路径 \
  --act_scales_path /输出/act_scales.pt \
  --smooth \
  --alpha 0.85 \
  --quantize
```

`--alpha 0.85` 只是 LLaMA-2-7B 等部分官方脚本的示例值。官方 README 中 LLaMA、Mistral、Mixtral、Falcon 的推荐值约为 0.6–0.9，复现时必须按模型保存。

## 关键设置

| 参数 | 官方观察 | 中文说明 |
|---|---:|---|
| weight bits | 8 | 官方 fake-quant 主路径为 INT8 weight |
| activation bits | 8 | 官方主设置为 dynamic activation W8A8 |
| calibration samples | 512 | Pile validation 随机句子 |
| `alpha` | 模型相关 | 控制量化难度从 activation 向 weight 迁移的强度 |

## 复现边界

- 官方代码的核心结论是 W8A8，不把低位宽 W4A4 的本地扩展写成 SmoothQuant 论文结果。
- fake quant 精度结果与 CUTLASS/FasterTransformer 的真实 INT8 性能结果分开记录。
