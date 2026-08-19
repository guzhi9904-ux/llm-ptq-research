# FlexRot-FP4 实验说明

## Phase 6：为什么做 W16A4 activation-only

W16A4 保持权重为 BF16，只量化 activation。这样可以先回答“旋转是否改善 activation 的 FP4 可量化性”，避免 weight quantization、GPTQ 补偿和 sequential drift 同时出现。Phase 6 得到的是机制定位，不是最终部署结论；因此 activation-only 最优角度不能直接当成 W4A4 最优角度。

## Phase 7：为什么做 W4A4 angle landscape

部署目标是 W4A4，weight 与 activation 会共同响应同一成对旋转。Phase 7 在固定网格
`0, 5.625, 11.25, 16.875, 22.5, 33.75, 45` 上先用 RTN 扫描，再把冻结角度放入 vanilla GPTQ 与强 MR-GPTQ。所有 196 个 Qwen2.5-1.5B decoder Linear 使用统一角度，禁止按层微调或看过 test PPL 后细搜。

## Phase 8：为什么做跨模型验证

单模型结果可能来自特定宽度、层数、权重分布或 tokenizer。Phase 8 分别检查 Qwen2.5 的规模扩展、Qwen3 的代际变化、Llama 的模型族变化。目标是验证“FP4 format 对 rotation strength 的偏好不同”这一冻结定性结论，而不是重新寻找并宣传 universal angle。

## 术语

- RTN：round-to-nearest。权重按当前 FP4 scale 直接舍入，不用 Hessian 做误差传播。
- GPTQ：用校准激活的 Gram/Hessian 近似逐列量化权重，并把当前列误差补偿到尚未量化的列。vanilla 配置关闭 Static ActOrder 与 MSE-grid。
- MR-GPTQ：本项目强基线。rotation、activation calibration、Static ActOrder、FP4-aware GPTQ、MSE-grid 与 sequential deployment calibration 组合运行。
- NVFP4：E2M1 元素、block16、E4M3 local scale、FP32 global scale。activation global scale 必须由独立 calibration 数据确定。
- MXFP4：E2M1 元素、block32、E8M0 scale。每块 scale 是 2 的幂，不使用 tensor-wide global scale。

## 角度含义

- 0°：正式实验中的 no rotation 基线。数学族的 `F(0)` 是符号对角，但不作为该 arm 的实际实现。
- 5.625°：Phase 7 固定网格上的弱旋转点；它是 Qwen2.5-1.5B NVFP4 W4A4 的当前最优点，不代表 universal。
- 45°：`F(45°)=H2`，tensor product 后严格等价于 native normalized Hadamard；表示 full mixing。

## MR-GPTQ 固定流程

```text
输入 BF16 模型
→ 构建 rotation
→ transform X/W
→ activation calibration
→ Static ActOrder
→ FP4-aware GPTQ
→ MSE-grid
→ sequential deployment calibration
→ W4A4 forward
→ WikiText-2 PPL
```

MSE-grid 与 rotation strength 是两个独立变量：前者在给定坐标系内搜索 scale，后者改变坐标系。任何消融都必须分别记录两者。

## 可比较性约束

- calibration 使用 WikiText-2 train，evaluation 使用 test。
- Phase 7 使用 8×512 窗口，PPL 预测 token 数必须是 `8×(512-1)=4088`；该结果现为历史探索协议，不与 paper-aligned 主表合并。
- paper-aligned calibration 使用 FineWeb-Edu 2048-token 序列，pilot/paper 分别为 128/1024 条并使用多个 seed。
- paper baseline 固定 0°/45°，并补 MMLU-CoT、GSM8K、HellaSwag、WinoGrande；baseline 完整后再扫描弱角度。
- NVFP4 每个 angle 都重新从相同 calibration token 计算 activation global scale。
- 同一对比中只允许改变声明的变量；模型 dtype、damping、traversal、ActOrder、MSE-grid 和评测窗口保持不变。
