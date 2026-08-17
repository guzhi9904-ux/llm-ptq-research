# QuaRot

## 方法定位

QuaRot 是 NeurIPS 2024 的端到端 W4A4KV4 旋转量化方法。官方代码为 [spcl/QuaRot](https://github.com/spcl/QuaRot)，本地干净版本固定在 `5008669b08c1f11f9b64d52d16fddd47ca754c5a`。

## 核心逻辑

1. 对 residual stream 应用保持模型函数不变的全局正交旋转。
2. 把可离线融合的旋转合并到 LayerNorm 和 Linear weight。
3. 对 down projection、attention output 等位置使用在线 Hadamard transform。
4. 同时量化 weight、activation、K cache 和 V cache。
5. weight 支持 RTN 或 GPTQ；activation/K/V 使用相应的 per-token/group quantizer。

## 关键实现

- 主入口：`fake_quant/main.py`
- 模型旋转：`fake_quant/rotation_utils.py`
- Hadamard：`fake_quant/hadamard_utils.py`
- 量化器：`fake_quant/quant_utils.py`
- GPTQ：`fake_quant/gptq_utils.py`

## 核心实验：LLaMA-2-7B W4A4KV4

```bash
# --rotate 开启函数保持旋转；--w_clip 对权重做 clipping 搜索。
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --rotate \
  --w_bits 4 \
  --a_bits 4 \
  --k_bits 4 \
  --v_bits 4 \
  --w_clip
```

命令需从 `QuaRot/fake_quant/` 目录运行。

## 参数说明

| 参数 | 中文说明 |
|---|---|
| `--rotate` | 应用 QuaRot 的全局和在线旋转 |
| `--w_bits/--a_bits` | weight 与 activation 位宽 |
| `--k_bits/--v_bits` | KV cache 中 K 与 V 的位宽 |
| `--w_clip` | 搜索更合适的 weight clipping 范围 |
| `--*_asym` | 对对应张量使用非对称量化 |
| `--*_groupsize` | 对应张量的 group size；未显式设置时必须记录代码默认值 |
| `--cal_dataset` | GPTQ 的校准数据；随机 RTN rotation 本身不需要学习 |

## 复现边界

- 本地 fake-quant 路径主要支持 LLaMA-2，迁移到 Qwen 时需要模型映射与等价性测试。
- `e2e/` 真实 kernel 和 `fake_quant/` 精度模拟是两条独立证据链。
