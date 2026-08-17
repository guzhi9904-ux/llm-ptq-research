# FlatQuant

## 方法定位

FlatQuant 是 ICML 2025 的 W4A4/W4A4KV4 方法，通过每个 Linear 的快速可学习 affine transformation 改善 weight 与 activation 的平坦程度。官方代码为 [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant)。

本地 commit 为 `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`，但工作树有本地 Qwen/部署修改，正式迁移前必须保存差异。

## 核心逻辑

1. 在每个 Linear 周围插入保持全精度函数等价的结构化 affine transform。
2. 使用 Kronecker/block 结构降低完整矩阵变换的训练和推理成本。
3. 可叠加 learnable diagonal scale、LWC 和 LAC。
4. 用 layer output MSE 优化变换；weight 可选 RTN 或 GPTQ。
5. 把可融合部分并入 weight，在线部分由专用 kernel 执行。

## 核心实验：LLaMA-3-8B W4A4KV4

```bash
# K/V 使用非对称 4 bit、group size 128；优化 15 epoch。
python main.py \
  --model ./modelzoo/meta-llama/Meta-Llama-3-8B \
  --w_bits 4 \
  --a_bits 4 \
  --k_bits 4 \
  --k_asym \
  --k_groupsize 128 \
  --v_bits 4 \
  --v_asym \
  --v_groupsize 128 \
  --cali_bsz 4 \
  --epoch 15 \
  --flat_lr 5e-3 \
  --lwc \
  --lac \
  --cali_trans \
  --add_diag \
  --output_dir ./outputs \
  --save_matrix \
  --lm_eval \
  --lm_eval_batch_size 16
```

## 参数说明

| 参数 | 官方示例 | 中文说明 |
|---|---:|---|
| `w_bits/a_bits` | 4/4 | weight 与 activation 位宽 |
| `k_bits/v_bits` | 4/4 | KV cache 位宽 |
| `k_groupsize/v_groupsize` | 128 | K/V 分组大小 |
| `cali_bsz` | 4 | calibration optimization batch size |
| `epoch` | 15 | affine transform 优化轮数 |
| `flat_lr` | `5e-3` | transform 学习率 |
| `lwc/lac` | 开启 | learnable weight/activation clipping |
| `cali_trans` | 开启 | 优化 calibration transform |
| `add_diag` | 开启 | 叠加 learnable diagonal scale |

## 重要语义

- 本地代码核查显示 activation 使用 per-token，weight 使用 per-channel；activation group quantization 不是该版本的通用实现。
- `--reload_matrix` 评测预训练矩阵和从头 calibration 是两条不同实验路径。
- fake-vLLM、real quant 和论文早期 kernel 版本需要分别固定依赖。
