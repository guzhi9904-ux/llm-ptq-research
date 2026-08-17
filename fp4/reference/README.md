# FP4 PTQ 独立参考实现

本目录根据论文公开公式独立编写 MR-GPTQ 与 MicroMix 的 PyTorch 数值参考实现，没有复制 FP-Quant 或 MicroMix 官方仓库的源文件。代码采用本目录的 MIT License；论文、名称和官方实现仍归原作者所有。

## 实现范围

MR-GPTQ 参考路径包含：

- MXFP4 E2M1、32 元素 block 与 E8M0 scale 近似。
- NVFP4 E2M1、16 元素 block、E4M3 group scale 与 tensor scale 近似。
- minmax/MSE scale search 和 MXFP scale range fitting。
- block-wise normalized Hadamard transform。
- 固定原始 quantization grid 后的 static ActOrder。
- 基于校准 Gram/Hessian 的逐列 GPTQ error compensation。
- identity、rotation + RTN、identity + GPTQ 和 MR-GPTQ 可控基线。

MicroMix 参考路径包含：

- MXFP4 E2M1、MXFP6 E3M2、MXFP8 E5M2/E4M3 网格。
- 论文阈值公式 `T(n)`。
- calibration activation 的 channel-wise absolute mean。
- channel 升序 permutation 与层级自适应 `p4/p6/p8`。
- activation 和对应 weight channel 的相同精度 fake quant。
- 平均元素位宽与 E8M0 scale 存储开销计算。

## 明确不包含

- QuTLASS、CUTLASS 或 MicroMix Blackwell CUDA kernel。
- FP4 bit packing、硬件特殊值和逐位一致性声明。
- Hugging Face 全模型自动替换、PPL 或 lm-eval 结果。
- 官方 checkpoint、官方代码或论文性能数字的复制。

因此该实现状态是 `synthetic-smoke-tested`，不是 `reproduced`，不能用 CPU fake quant 延迟替代真实 GPU 性能。

## 安装与测试

```bash
cd fp4/reference
pip install -e .
pytest -q

python examples/synthetic_demo.py mr-gptq
python examples/synthetic_demo.py micromix
```

## 对已提取 Linear 张量运行

输入文件应由 `torch.save(tensor, path)` 生成：

```bash
ptq-fp4-reference mr-gptq \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --format mxfp4 \
  --hadamard-group-size 128 \
  --output mr_gptq_result.pt

ptq-fp4-reference micromix \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --channel-alignment 32 \
  --output micromix_result.pt
```

## 论文对应与参考选择

- MR-GPTQ 论文：[Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202)。
- MicroMix 论文：[MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models](https://arxiv.org/abs/2508.02343)。
- E2M1 使用正级别 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`。
- MicroMix 论文没有完全规定如何把所有 calibration 元素的阈值标签汇总成整数 channel 数；本实现公开采用“元素比例平均 + largest remainder”的规则，并允许 channel alignment。该选择必须与官方结果分开标注。
- MSE scale search 使用可读的有限网格搜索，目标是验证算法关系，不是复刻官方高性能实现。

独立实现和论文公式的逐项对应见 `IMPLEMENTATION_NOTES.md`。
