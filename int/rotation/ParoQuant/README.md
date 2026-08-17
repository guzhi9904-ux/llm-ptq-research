# ParoQuant

## 方法定位

ParoQuant 对应预印本《Pairwise Rotation Quantization for Efficient Reasoning LLM Inference》（arXiv:2511.10645）。官方代码为 [z-lab/paroquant](https://github.com/z-lab/paroquant)，本地 main 固定在 `f74a96c306f7db3938e758d7dbc945d450f9f23f`。

当前不能把它标成 ICLR/ICML/NeurIPS 已录用论文。

## 核心逻辑

1. 把大旋转矩阵分解为独立、硬件友好的 pairwise Givens rotations。
2. 学习 rotation angle，压制 weight outlier。
3. 配合 channel-wise scaling 缩小 quantization group 内动态范围。
4. 主要目标是 INT4 weight-only，并保持接近 AWQ 的推理效率。

## 代码版本注意

本地盘点显示当前 main 更偏向 vLLM、Transformers 与 MLX 部署。论文复现需要确认 README 指向的 legacy branch，不能直接用当前产品化 CLI 代替论文实验。

## 核心实验计划

- 固定 legacy reproduction branch 和 commit。
- 复现论文使用的 reasoning 模型与任务。
- 对比 RTN、AWQ/GPTQ 与 ParoQuant INT4 weight-only。
- 单独消融 pairwise rotation 与 channel-wise scaling。
- 记录真实 kernel 速度，不用 fake quant 推断部署性能。

正式命令将在 legacy branch 核查后补入。
