# 本地修改与官方快照差异

核查日期：2026-08-17。所有 `upstream/` 均从 Git commit 导出，没有复制下列工作树修改。

## OmniQuant

- 官方快照：`origin/main@feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`。
- 本地分支：`agent/lensq-omniquant-smoke@3a7d4fbc174981b4fd5d84ddd268e048d9c654ca`。
- 本地分支相对官方 main：领先 9 个 commit。
- 工作树另有 28 项修改或未跟踪文件。
- 主要内容：LensQ/tuned-lens 目标、Qwen2 支持、轨迹诊断、OPT/Qwen 实验脚本和本地模型输出。
- 处理：论文代码使用官方 main；LensQ/Qwen 研究后续只挑选必要部分迁入 `experiments/`，不放进 OmniQuant `upstream/`。

## FlatQuant

- 官方快照：`origin/main@9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`。
- Git commit 与 origin 一致，但工作树有 35 项修改或未跟踪文件。
- tracked diff：21 个文件，约 855 行新增、92 行删除。
- 主要内容：Qwen2.5 模型适配、KV cache、低显存 kernel、非 2 次幂 `o_proj`、OmniSQL/BIRD 评测与部署脚本。
- 处理：官方论文代码保持干净；部署优化和 OmniSQL 实验后续按功能拆到 `experiments/adapters/flatquant/`。

## FP-Quant / MR-GPTQ

- 官方快照：`origin/master@d2e3092f968262c4de5fb050e1aef568a280dadd`。
- 工作树有 5 个 tracked 文件修改，共约 25 行新增、10 行删除。
- 修改文件：`model_quant.py`、`src/quantization/gptq.py`、`quantizer.py`、`transforms.py`、`llama_utils.py`。
- 处理：`upstream/` 使用干净 commit；本地修正需要逐条审查后保存为最小 patch，不能静默覆盖官方 MR-GPTQ。

## SliderQuant

- 官方快照：`origin/main@eed0b8542f208c0e1d629ee3dd36f69416bb2a2e`。
- 没有 tracked 修改；有 2 组未跟踪输出/脚本。
- 内容：layer observation 的 smoke CSV/图片与复现实验脚本。
- 处理：论文源码保持干净；复现脚本迁入 `experiments/` 前先核查参数和数据来源。

## 不纳入版本控制的大文件

OmniQuant 未跟踪目录包含本地 `model.safetensors`。模型、checkpoint、activation cache、原始 outputs 和图片型运行产物不迁入本仓库，只保留配置、脚本、结果摘要和哈希。
