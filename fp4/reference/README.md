# FP4 PTQ 复现代码

这里放的是我们按论文公式自己写的 MR-GPTQ 和 MicroMix PyTorch 代码，没有直接搬 FP-Quant 或 MicroMix 仓库里的源码。代码使用本目录的 MIT License，方法名和论文仍属于原作者。

## 目前写了什么

MR-GPTQ 部分包括：

- MXFP4 E2M1、32 元素 block 与 E8M0 scale 近似。
- NVFP4 E2M1、16 元素 block、E4M3 group scale 与 tensor scale 近似。
- minmax/MSE scale search 和 MXFP scale range fitting。
- block-wise normalized Hadamard transform。
- 固定原始 quantization grid 后的 static ActOrder。
- 基于校准 Gram/Hessian 的逐列 GPTQ error compensation。
- identity、rotation + RTN、identity + GPTQ 和 MR-GPTQ 几组基线。

MicroMix 部分包括：

- MXFP4 E2M1、MXFP6 E3M2、MXFP8 E5M2/E4M3 网格。
- 论文阈值公式 `T(n)`。
- calibration activation 的 channel-wise absolute mean。
- channel 升序 permutation 与层级自适应 `p4/p6/p8`。
- activation 和对应 weight channel 的相同精度 fake quant。
- 平均元素位宽与 E8M0 scale 存储开销计算。

## 目前还没做什么

- QuTLASS、CUTLASS 或 MicroMix Blackwell CUDA kernel。
- FP4 bit packing、硬件特殊值和逐 bit 对齐检查。
- Hugging Face 全模型自动替换、PPL 或 lm-eval 结果。
- 官方 checkpoint、官方代码或论文性能数字的复制。

目前代码只用合成张量跑通过，还没有做完整模型实验。CPU 上的 fake quant 运行时间也不能当作 GPU kernel 性能。

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

## 实现时采用的规则

- MR-GPTQ 论文：[Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202)。
- MicroMix 论文：[MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models](https://arxiv.org/abs/2508.02343)。
- E2M1 使用正级别 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`。
- MicroMix 论文没有写清楚怎样把所有 calibration 元素的阈值标签换成整数 channel 数。这里先对元素比例取平均，再用 largest remainder 补齐 channel 数，也可以设置 channel alignment。这个处理可能和官方代码不同。
- MSE scale search 用有限网格逐个尝试，写法比较直观，但速度不是重点。

各段代码对应哪条论文公式，见 `IMPLEMENTATION_NOTES.md`。
