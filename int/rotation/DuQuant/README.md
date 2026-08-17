# DuQuant

## 方法定位

DuQuant 是 NeurIPS 2024 Oral，使用 Rotation Transformation 与 Permutation Transformation 分散 activation outlier。官方代码为 [Hsu1023/DuQuant](https://github.com/Hsu1023/DuQuant)，本地干净 commit 为 `d56cfc6fe97c34c0eb100fec82fe439865905679`。

## 核心逻辑

1. 识别 FFN `down_proj` 等位置的 massive outlier 与普通 outlier。
2. 用 block rotation 在局部通道内分散极值。
3. 用 permutation 重新组织通道，使量化 group 内分布更均衡。
4. 对 weight/activation 做 W4A4，可选 LWC、静态 weight clipping 和 activation clipping。

## 预处理

```bash
# 所有模型共用的 rotation 基础数据只需生成一次。
python get_rot.py

# 每个模型分别收集 activation scale 与 shift。
python generate_act_scale_shift.py --model /模型路径
```

## 核心实验：官方本地脚本设置

```bash
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --wbits 4 \
  --abits 4 \
  --block_size 128 \
  --max_rotation_step 256 \
  --epochs 0 \
  --lwc \
  --alpha 0.6 \
  --smooth \
  --lac 0.9 \
  --swc 0.8 \
  --eval_ppl \
  --task arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa
```

## 参数说明

| 参数 | 本地官方脚本值 | 中文说明 |
|---|---:|---|
| `block_size` | 128 | rotation matrix 的局部 block 大小 |
| `max_rotation_step` | 256 | greedy rotation 最大搜索步数 |
| `permutation_times` | 脚本未显式设置 | permutation 变换次数，需读取 argparse 默认值 |
| `alpha` | 0.6 | smoothing 强度 |
| `lac` | 0.9 | activation clipping ratio |
| `swc` | 0.8 | 非 LWC 路径的 static weight clipping ratio |
| `lwc` | 开启 | 使用 learnable weight clipping |
| `epochs` | 0 | 当前脚本不训练 LWC；与论文训练设置需再次对照 |

`epochs=0` 与 `--lwc` 同时出现需要在逻辑核查时确认究竟是加载参数、仅启用量化边界，还是脚本遗留设置。
