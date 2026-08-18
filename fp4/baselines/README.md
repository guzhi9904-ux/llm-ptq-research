# FP4 基线

FP4 不能只写成一个笼统的 4 bit 设置。MXFP4 使用 32 元素 block 和 E8M0 scale；NVFP4 使用 16 元素 block、E4M3 block scale 和 FP32 tensor scale。因此下面三条基线都要分别跑两种格式。

## 三条基础对照

### RTN

直接按格式网格和 absmax scale 舍入，不使用 Hessian，也不旋转。它主要检查 E2M1 网格、block 划分和 scale 层级有没有写错。

### Rotation + RTN

先在输入通道上做 block Hadamard，再用和 RTN 完全相同的量化器。MR-GPTQ 论文的基础设置让 Hadamard group 与量化 block 对齐：MXFP4 为 32，NVFP4 为 16。

### GPTQ

weight 在 FP4 网格上做逐列二阶误差补偿，activation 仍使用 RTN。基础 GPTQ 使用 absmax scale、默认列顺序和 `relative_damp=0.01`，不加入 rotation、MSE scale 或 static ActOrder。

这三项和 MR-GPTQ 的区别如下：

| 设置 | RTN | Rotation + RTN | GPTQ | MR-GPTQ |
|---|---:|---:|---:|---:|
| Hadamard | 否 | 是 | 否 | 是 |
| GPTQ 补偿 | 否 | 否 | 是 | 是 |
| MSE scale | 否 | 否 | 否 | 是 |
| static ActOrder | 否 | 否 | 否 | 是 |

## 可运行代码

张量级 PyTorch 入口在 [`../reference/src/ptq_fp4_reference/baselines.py`](../reference/src/ptq_fp4_reference/baselines.py)，六组完整参数在 [`configs/baseline_matrix.yaml`](configs/baseline_matrix.yaml)。例如：

```bash
cd fp4/reference
pip install -e .

# MXFP4 RTN
ptq-fp4-reference baseline \
  --baseline rtn \
  --format mxfp4 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output mxfp4_rtn.pt

# NVFP4 Hadamard + RTN
ptq-fp4-reference baseline \
  --baseline rotation-rtn \
  --format nvfp4 \
  --hadamard-group-size 16 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output nvfp4_rotation_rtn.pt

# MXFP4 GPTQ
ptq-fp4-reference baseline \
  --baseline gptq \
  --format mxfp4 \
  --relative-damp 0.01 \
  --weight weight.pt \
  --activations calibration_inputs.pt \
  --output mxfp4_gptq.pt
```

完整模型实验可直接使用 [`../methods/MR-GPTQ/upstream/model_quant.py`](../methods/MR-GPTQ/upstream/model_quant.py)。张量级代码用于核对算法和参数，不代表 Blackwell kernel 的真实速度。
