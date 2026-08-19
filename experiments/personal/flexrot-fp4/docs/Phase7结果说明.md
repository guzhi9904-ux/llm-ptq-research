# Phase 7 结果说明

模型：Qwen2.5-1.5B-Instruct；设置：W4A4；数据：WikiText-2；评测：test split 前 8 个连续 512-token 窗口，共预测 4088 token。

## RTN angle landscape 的关键点

| Format | θ | PPL |
|---|---:|---:|
| NVFP4 | 0° | 15.547818 |
| NVFP4 | 5.625° | 15.301589 |
| NVFP4 | 45° | 16.123654 |
| MXFP4 | 0° | 20.277765 |
| MXFP4 | 33.75° | 17.438706 |
| MXFP4 | 45° | 17.835033 |

NVFP4 在固定网格上选择 5.625°；MXFP4 的强 mixing 区域显著优于 0°。这支持“format-specific strength”，不支持“5.625° universal”。

## 替换实验

| Format | Pipeline | 弱/近全角 PPL | 45° PPL | Δ vs 45° |
|---|---|---:|---:|---:|
| NVFP4 | GPTQ | 15.167781（5.625°） | 15.207757 | -0.039977 |
| NVFP4 | MR-GPTQ | 14.848294（5.625°） | 15.019048 | -0.170754 |
| MXFP4 | MR-GPTQ | 16.541120（33.75°） | 16.278792 | +0.262328 |

vanilla GPTQ 关闭 Static ActOrder 与 MSE-grid；MR-GPTQ 两者均启用，并保持 1% damping、128 列 traversal、FP4-aware projector 与 sequential deployment calibration 不变。每一对比较只改变 θ。

## 上传门禁

`python scripts/run_phase7.py` 会用新仓库代码运行五个必需 arm；`--check-only` 只检查新仓库结果目录。相对误差容限为 0.1%，且每个输出必须记录 4088 predicted tokens。

2026-08-12 已使用本仓库代码在 RTX 3060 Ti、PyTorch 2.5.1/CUDA 环境完成门禁：

| Arm | 新仓库 PPL | 目标 PPL | 状态 |
|---|---:|---:|---|
| NVFP4 RTN 0° | 15.5478176280 | 15.547818 | 通过 |
| NVFP4 RTN 5.625° | 15.3015887612 | 15.301589 | 通过 |
| NVFP4 RTN 45° | 16.1236544184 | 16.123654 | 通过 |
| NVFP4 MR-GPTQ 5.625° | 14.8482942670 | 14.848294 | 通过 |
| NVFP4 MR-GPTQ 45° | 15.0190479993 | 15.019048 | 通过 |

五项均预测 4088 tokens；最大相对差远小于 0.1% 容限。原始运行 JSON、逐层 checkpoint 和 gate JSON 位于被 `.gitignore` 排除的 `results/`，不会上传 GitHub。
