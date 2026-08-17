# 第三方代码许可证核查

核查日期：2026-08-17。本文件只记录源码仓库中可见的许可证，不构成法律意见。

## 核查结果

| 方法 | 根目录许可证 | 当前处理 |
|---|---|---|
| GPTQ | Apache-2.0 | 保留原 LICENSE，可在满足条款时再分发 |
| SmoothQuant | MIT | 保留原 LICENSE |
| OmniQuant | MIT，文件名为 `LICENCE` | 保留原文件名和全文 |
| AffineQuant | Apache-2.0 | 保留原 LICENSE |
| QuaRot | Apache-2.0 | 核心源码已导入；三个 submodule 未复制，需分别遵守其许可证 |
| DuQuant | MIT | 保留原 LICENSE |
| SpinQuant | CC BY-NC 4.0 | 仅限非商业使用；不得当作 MIT/Apache 代码混合授权 |
| OSTQuant | Apache-2.0 | 保留原 LICENSE |
| FlatQuant | MIT | 核心源码已导入；三个 submodule 未复制，需分别遵守其许可证 |
| GuidedQuant | MIT | 根目录为 MIT；仓库内附带组件还有各自 LICENSE |
| Q-Palette | MIT | 保留原 LICENSE |
| ParoQuant legacy | MIT | 使用论文复现 legacy commit，保留原 LICENSE |
| SliderQuant | Apache-2.0 | 保留原 LICENSE |
| DuQuant++ | MIT | 保留原 LICENSE |
| FP-Quant / MR-GPTQ | **未发现根目录 LICENSE** | 本地快照已由 `.gitignore` 排除，不进入可发布 Git 历史 |
| MicroMix `micromix` | **未发现根目录 LICENSE** | 本地快照已由 `.gitignore` 排除，不进入可发布 Git 历史 |
| MixFP4 | 官方代码未定位 | 不导入源码 |

## 本仓库自己编写的部分

`fp4/reference/` 是我们按论文公式自己写的代码，使用该目录内的 MIT License。其中没有 FP-Quant、MicroMix 或 QuTLASS 的源文件。方法名称只用来说明这段代码在复现哪篇论文。

## Submodule 处理

QuaRot 和 FlatQuant 的源码树引用：

- NVIDIA CUTLASS。
- NVIDIA NVBench。
- Dao-AILab fast-hadamard-transform。

MicroMix 引用 NVIDIA CUTLASS。

固定的 submodule commit：

| 方法 | 路径 | commit |
|---|---|---|
| QuaRot | `third-party/cutlass` | `ffa34e70756b0bc744e1dfcc115b5a991a68f132` |
| QuaRot | `third-party/fast-hadamard-transform` | `4ea722e434e3d4f2a14522341959ebdbe62be2de` |
| QuaRot | `third-party/nvbench` | `d8dced8a64d9ce305add92fa6d274fd49b569b7e` |
| FlatQuant | `third-party/cutlass` | `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e` |
| FlatQuant | `third-party/fast-hadamard-transform` | `d1a56eee9d502e67faacf61b7b947180d66b32a0` |
| FlatQuant | `third-party/nvbench` | `a171514056e5d6a7f52a035dd6c812fa301d4f4f` |
| MicroMix | `cutlass` | `a1aaf2300a8fc3a8106a05436e1a2abad0930443` |

当前 `upstream/` 是固定 commit 的 `git archive`，没有递归复制这些大型依赖。`.gitmodules` 已保留，用于记录 URL；精确 submodule commit 另写入来源记录。这样既避免仓库体积膨胀，也避免把第三方依赖错误地归入论文代码许可证。

## 公开发布前的门槛

1. 不给 `upstream/` 添加统一仓库许可证声明。
2. 每个方法保留原 LICENSE、NOTICE 和引用信息。
3. SpinQuant 目录显著标记“仅非商业使用”。
4. FP-Quant 与 MicroMix 在没有明确许可证时只保留固定来源说明和本地获取步骤，不随主仓库公开发布源码。
5. 模型权重、数据集和预训练量化参数需要单独核查许可证。
