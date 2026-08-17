# 本地实验代码迁移队列

本文件记录需要从旧工作树迁入 `experiments/` 的内容。迁移时只选择本次实验真正需要的代码，不复制模型、缓存和无关输出。

## 第一优先级：FP4 统一实验

- `quantization_harness/` 中格式正确的 INT、MXFP4、NVFP4 quantizer。
- `flexrot-fp4/` 中 RTN、GPTQ、MR-GPTQ 与可调 Hadamard strength runner。
- 现有 WikiText-2 calibration/evaluation 数据隔离逻辑。
- format endpoint、scale hierarchy、rotation equivalence 与 GPTQ tests。

目标位置：`experiments/adapters/`、`experiments/runners/`、`experiments/tests/`。

## 第二优先级：论文实现适配

- OmniQuant：只迁移 Qwen2 adapter 和必要的 dtype/model-loading 修正；LensQ objective 独立成实验方法。
- FlatQuant：只迁移 Qwen2.5 model mapping、必要 kernel fix 和评测 adapter；OmniSQL/BIRD 作为独立实验组。
- FP-Quant：逐条审查 5 个 tracked 修改，分别判断是 bug fix、Qwen 适配还是研究改动。
- SliderQuant：迁移 layer observation 脚本，不迁移 smoke 图片和临时输出。

## 明确不迁移

- `model.safetensors`、checkpoint、activation cache。
- 无配置来源的旧日志。
- 与论文代码无关的临时 notebook、build 目录和二进制产物。
- 不能说明来源或许可证的第三方源码副本。
