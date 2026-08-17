# Q-Palette

## 方法定位

Q-Palette 是 NeurIPS 2025 的 weight-only PTQ 方法，针对旋转后近似 Gaussian 的 weight 设计 fractional-bit quantizer，并在资源约束下联合选择量化方案与 layer fusion。官方 `main@025f27f311774d4618fcc0d47e4c8d33ee568a6c` 已导入 `upstream/`，许可证为 MIT。

## 核心逻辑

1. 先利用 rotation 等方法把 weight 分布 Gaussianize。
2. 提供 trellis-coded、vector 和 scalar 等不同复杂度的 fractional-bit quantizer。
3. 估计每层选择不同 quantizer 的 distortion/sensitivity。
4. 在平均 bit、内存或 throughput 约束下求解 mixed-scheme quantization（MSQ）。
5. latency-aware 版本把 layer fusion 决策与 quantizer selection 联合优化。

## 官方机器设置

- GPU：NVIDIA RTX 4090。
- CPU：AMD EPYC 7B13 64-Core。
- 系统：Ubuntu 22.04.5。
- CUDA：12.4。

## 核心实验一：内存约束 MSQ

```bash
# 目标平均 bit 为 3.25；输出每层量化方案 qdict。
python solve_mem_const.py \
  --model meta-llama/Llama-3.1-8B \
  --target_bitwidth 3.25

# 用生成的 qdict 评测 WikiText-2 PPL。
python eval_qdict.py \
  --qdict_path msq_results/3_8b/mem_constrained/default/3.25bit.pt
```

## 核心实验二：吞吐约束 MSQ

```bash
# 在 RTX 4090、batch size 1 下把目标吞吐设为 200 token/s。
python solve_lat_const.py --target_thp 200
```

关键开关：

- `--use_cc`：允许 CUDA-Core 实现。
- `--no_fuse`：关闭 fusion-aware MSQ，用于消融 layer fusion 的贡献。

## 核心实验三：速度测量

```bash
python eval/measure_latency_merge_simt.py \
  --hf_path meta-llama/Llama-3.1-8B \
  --qdict_path /量化配置.pt \
  --use_inc_mlp \
  --use_inc_attn \
  --merge_info_path /融合配置.pt \
  --print_result
```

## 复现边界

- 这是 weight-only fractional-bit 路线，不归入统一 W4A4 主表。
- 吞吐约束与具体 RTX 4090 kernel 强相关，跨 GPU 时必须重新测量 cost model。
