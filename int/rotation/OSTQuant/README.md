# OSTQuant

## 方法定位

OSTQuant 是 ICLR 2025 方法，正式名称为《OSTQuant: Refining Large Language Model Quantization with Orthogonal and Scaling Transformations for Better Distribution Fitting》。官方代码为 [BrotherHappy/OSTQuant](https://github.com/BrotherHappy/OSTQuant)。

当前状态：`source-pinned`。官方 `main@ab64362da147291612d077accaab5d3ed7b508b6` 与 Apache-2.0 LICENSE 已导入 `upstream/`。

## 核心逻辑

1. 用 Quantization Space Utilization Rate（QSUR）评价变换后数据对量化空间的利用。
2. 对 residual stream 学习 global orthogonal transformation。
3. 在 attention 和 FFN 路径学习 scaling transformation。
4. 将成对变换放到函数等价位置，并把可融合部分并入 weight。
5. 使用 KL-Top loss 减少有限校准数据下的优化噪声，并保留更重要的语义信息。

## 核心实验：LLaMA-2-7B W4A16KV16

官方从头复现入口为四卡脚本：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
sh scripts/w4a16kv16.sh
```

脚本的关键设置：

```text
model=Llama-2-7b-hf
loss_type=kl_top
max_steps=100
per_device_train_batch_size=4
a_bits=16, k_bits=16, v_bits=16, down_bits=16
train_enable_wquant=True
bf16=True
lm_eval=True
```

还应保留：

- LLaMA-3-8B W4A4KV4 主结果。
- orthogonal-only、scaling-only 与完整 OSTQuant 消融。
- QSUR 与量化后 PPL/zero-shot 的对应关系。
- KL-Top 与普通 MSE/KL 目标对照。

## 仍需核查的参数

- 校准数据集、样本数、序列长度和 seed；顶层 README 未完整说明这些值。
- orthogonal/scaling 初始化、学习率、step/epoch。
- KL-Top 的 top-k/温度与损失权重。
- weight、activation、KV 的位宽、group size、对称性和 clipping。
- weight 使用 RTN 还是 GPTQ，以及表格中的对应关系。
