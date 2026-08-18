# FP4 张量级参考代码

这里放便于读和测试的 PyTorch 实现，用来检查 FP4 网格、scale、rotation 和误差补偿。论文仓库原代码放在 `fp4/methods/*/upstream/`，两者不混在一起。

## 已有内容

- MXFP4：E2M1、32 元素 block、E8M0 scale 近似。
- NVFP4：E2M1、16 元素 block、E4M3 block scale 和 tensor scale。
- 基线：RTN、Hadamard + RTN、普通 GPTQ，MXFP4/NVFP4 都能运行。
- MR-GPTQ：MSE scale、block Hadamard、static ActOrder 和 GPTQ error compensation。
- MicroMix：MXFP4/6/8、activation channel 排序和 `p4/p6/p8` 分区。
- MixFP4：E2M1/E1M2 双格式、E4M3 scale 和逐 block MSE 选择。

## 安装和测试

```bash
cd fp4/reference
pip install -e .
pytest -q
```

测试只用小张量，不下载模型。它可以检查算法路径，但不能代替 PPL、下游任务和 kernel 测速。

## FP4 baseline

```bash
ptq-fp4-reference baseline \
  --baseline rtn \
  --format mxfp4 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output mxfp4_rtn.pt

ptq-fp4-reference baseline \
  --baseline rotation-rtn \
  --format nvfp4 \
  --hadamard-group-size 16 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output nvfp4_rotation_rtn.pt

ptq-fp4-reference baseline \
  --baseline gptq \
  --format mxfp4 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output mxfp4_gptq.pt
```

三条基线的固定区别和六组配置见 [`../baselines/README.md`](../baselines/README.md)。

## 论文方法

```bash
# MR-GPTQ：默认打开 rotation、MSE scale、static ActOrder 和 GPTQ
ptq-fp4-reference mr-gptq \
  --format mxfp4 \
  --hadamard-group-size 128 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output mr_gptq.pt

# MicroMix：按 calibration activation 分配 MXFP4/6/8 channel
ptq-fp4-reference micromix \
  --channel-alignment 32 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output micromix.pt

# MixFP4 Algorithm 1：逐 16 元素 block 在 E2M1/E1M2 中选择
ptq-fp4-reference mixfp4 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output mixfp4.pt
```

## 目前没有覆盖的部分

- FP4 bit packing 和 Blackwell Tensor Core 的逐 bit 对齐。
- MR-GPTQ 的 QuTLASS 部署、MicroMix mixed-precision GEMM 和 MixFP4 E2M2 硬件改动。
- Hugging Face 全模型自动替换和论文表格的完整复跑。

真实 kernel 和全模型入口请使用对应 `upstream/`。大型外部依赖可在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File fp4/fetch_dependencies.ps1 -Method All
```

代码和论文公式的对应关系见 [`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md)。
