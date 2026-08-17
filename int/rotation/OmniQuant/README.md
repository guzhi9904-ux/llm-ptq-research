# OmniQuant

## 方法定位

OmniQuant 是 ICLR 2024 Spotlight，核心组件是 Learnable Equivalent Transformation（LET）与 Learnable Weight Clipping（LWC）。官方代码为 [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant)。

本地版本在非官方研究分支且有修改，因此当前只作为代码逻辑参考，不作为干净论文快照。

## 核心逻辑

1. 按 Transformer block 顺序缓存校准输入。
2. 以全精度 block 输出为重构目标。
3. LWC 学习 weight clipping 边界。
4. LET 学习 LayerNorm-to-Linear、Linear-to-Linear 以及 Q/K 等位置的等价 scale/shift。
5. 冻结原模型权重，只优化量化相关参数。

## 核心实验：LLaMA-7B W4A4

```bash
# 20 epoch 优化 LET 与 LWC；同时评测 PPL 和六个 zero-shot 任务。
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model /模型路径/llama-7b \
  --epochs 20 \
  --output_dir ./log/llama-7b-w4a4 \
  --eval_ppl \
  --wbits 4 \
  --abits 4 \
  --lwc \
  --let \
  --tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

## 参数说明

| 参数 | 官方默认或示例 | 中文说明 |
|---|---:|---|
| `epochs` | 20（README 示例） | block 级量化参数优化轮数；设为 0 可直接加载预训练参数评测 |
| `nsamples` | 128 | 校准样本数 |
| `lwc_lr` | `1e-2` | LWC 参数学习率 |
| `let_lr` | `5e-3` | LET 参数学习率 |
| `group_size` | 未设置时 per-channel | weight group size |
| `seed` | 本地代码核查为 2 | 数据采样 seed；最终需与官方脚本逐项对照 |
| `sequence_length` | 本地代码核查为 2048 | 校准序列长度 |

## 重要语义

- 官方 quantizer 默认通常是带 zero-point 的非对称 affine quantization，不能与 `[-7,7]` 对称 INT4 直接合并。
- 本地分支包含非上游的 LensQ/Qwen 修改。迁移时重新获取干净 upstream，并把本地实验改动保存为独立 patch。
