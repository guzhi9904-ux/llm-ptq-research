# AffineQuant

## 方法定位

AffineQuant 是 ICLR 2024 的可学习等价仿射变换 PTQ 方法。官方代码为 [bytedance/AffineQuant](https://github.com/bytedance/AffineQuant)。

当前本地只有 `AffineQuant-main/` 源码快照，没有 Git 元数据；正式纳入前需要重新取得官方固定 commit。

## 核心逻辑

AffineQuant 把可优化范围从简单 per-channel scaling 扩展到更一般的等价 affine transformation，并在冻结模型权重的情况下，用校准数据优化变换与 clipping，使 weight/activation 的量化后 block 输出接近全精度输出。

## 核心实验：LLaMA-7B W4A4

以下命令来自官方 README，待固定 commit 后再次逐项核对：

```bash
# --use_matrix 开启 Q/K 等位置的 affine matrix；--aug_loss 开启增强重构损失。
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model /模型路径/llama-7b \
  --epochs 20 \
  --output_dir ./log/llama-7b-w4a4 \
  --eval_ppl \
  --wbits 4 \
  --abits 4 \
  --lwc \
  --let \
  --aug_loss \
  --use_matrix \
  --sf 0.1 \
  --tasks hendrycksTest,piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

## 参数说明

| 参数 | 官方示例或默认 | 中文说明 |
|---|---:|---|
| `epochs` | 20 | 仿射变换与 clipping 的优化轮数 |
| `nsamples` | 128 | 校准样本数 |
| `group_size` | 未设置时 per-channel | weight 量化分组 |
| `use_matrix` | W4A4 示例开启 | 使用 Q/K affine matrix |
| `use_ln_matrix` | weight-only 示例可开启 | 使用 LayerNorm 相关 affine matrix |
| `sf` | W4A4 示例为 0.1 | gradual mask 的稳定因子 |

## 待核查项

- 论文表格与官方脚本的模型、数据集、epoch 和任务设置是否完全一致。
- 本地快照与当前 official main 的差异。
- real quant 路径使用的 AutoGPTQ 版本和 kernel 语义。
