# MixFP4

MixFP4 在 NVFP4 的每个 16 元素 block 中选择 E2M1 或 E1M2。论文为 2026 年预印本《MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations》。截至 2026-08-18，没有找到作者明确发布的官方代码；同名第三方仓库没有论文作者说明，因此没有当作官方源码导入。

## 量化步骤

论文 Algorithm 1 给出的流程是：

1. 计算 tensor scale：`s32 = max(abs(X)) / 2688`。
2. 每个 block 为 E2M1 计算 E4M3 scale，最大幅值按 6 处理。
3. 同一个 block 为 E1M2 计算另一条 E4M3 scale，E1M2 乘 2 后按对称 INT4 的最大幅值 7 处理。
4. 分别量化和反量化，比较两条路径的 block MSE。
5. 保存误差更小的 4 bit payload 和 block scale。
6. 用 E4M3 scale 的 sign bit 保存格式类型，不增加额外 metadata。

可运行的 PyTorch 版本在 [`../../reference/src/ptq_fp4_reference/mixfp4.py`](../../reference/src/ptq_fp4_reference/mixfp4.py)，实现了上面的 RTN 和静态格式选择：

```bash
cd fp4/reference
pip install -e .
ptq-fp4-reference mixfp4 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output mixfp4.pt
```

## 论文实验设置

- 软件主实验的 block size 为 16。
- SmoothQuant：`wikitext-raw-v1` train，512 个样本，每个 512 token，`alpha=0.5`。
- GPTQ：使用 FP-Quant 实现，`wikitext-raw-v1` train，1024 个样本，sequence length 2048；格式在 GPTQ 前静态选好，误差补偿时不再改变。
- SpinQuant：100 个优化 step，每个 step 2048 token；后续 GPTQ 使用动态格式选择。
- 主要指标包括 WikiText perplexity 和下游准确率；硬件部分另外报告 E2M2 数据通路的面积和功耗开销。

## 还缺什么

当前代码没有论文提出的 E2M2 Tensor Core 改动，也没有作者 kernel。它可以检查 Algorithm 1 和 GPTQ 前的静态格式选择，不能复现论文硬件面积、功耗或真实吞吐。
