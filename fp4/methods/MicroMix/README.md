# MicroMix

## 方法定位

MicroMix 是 ICLR 2026 的 mixed-precision microscaling PTQ 方法，联合使用 MXFP4、MXFP6 和 MXFP8，并提供面向 Blackwell 的混合精度 GEMM kernel。官方代码为 [lwy2020/MicroMix](https://github.com/lwy2020/MicroMix)。

官方 `micromix@c57370bc38f999e1f75aada9c4b8a85baa44aaae` 已导入 `upstream/`。该分支根目录没有 LICENSE；本仓库按用户确认的上传授权保留源码，并在 `SOURCE.md` 中记录固定版本。

## 核心逻辑

1. 用 calibration activation 统计量评价 channel 敏感度。
2. 对 channel 重排，使不同精度 channel 能被 kernel 高效处理。
3. 为敏感 channel 分配 MXFP6/MXFP8，其余使用 MXFP4。
4. 保存 `reorder_indices`、`p6_num` 和 `p8_num`。
5. 用支持任意 MXFP4/6/8 channel 组合的 GEMM kernel 计算，并输出 BF16。

## 核心实验一：预处理

```bash
# 使用 32 条、长度 2048 的校准样本；按 mean activation metric 排序。
python reorder_indices.py \
  --model /模型路径 \
  --samples 32 \
  --seqlen 2048 \
  --act_sort_metric mean
```

| 参数 | 官方 README 值 | 中文说明 |
|---|---:|---|
| `samples` | README 命令为 32；argparse 默认 128 | 核心复现必须显式写 32，不能依赖默认值 |
| `seqlen` | 2048 | calibration 序列长度 |
| `act_sort_metric` | `mean` | channel 排序使用的 activation 统计量 |

## 核心实验二：精度评测

```bash
# test.sh 负责 zero-shot、few-shot 与 PPL；任务和 shot 数需从固定 branch 核查。
bash test.sh /模型路径
```

## 核心实验三：效率

```bash
# MicroMix：batch 8、prefill 长度 2048。
python benchmarks/benchmark_e2e_micromix.py \
  --model llama-3.1-8b \
  --batch_size 8 \
  --prefill_seq_len 2048
```

`mgemm/` 为真实 kernel 路径。必须记录 CUDA 12.8、GPU 型号、block/channel 分配、batch size 和 shape；不能用 Python fake quant 延迟代替。

## 待核查项

- `p6_num/p8_num` 的选择约束、平均 bit 预算和每层分配规则。
- `test.sh` 的模型、数据集、任务、shot、seed 和 lm-eval 版本。
- kernel 的实际输入布局与 scale 编码。

## 本仓库里的复现代码

官方源码已经放在 [`upstream/`](upstream/)：`reorder_indices.py` 负责离线 channel 排序，`mgemm/` 是 mixed-precision CUDA kernel，`benchmarks/` 是效率测试。CUTLASS 体积较大，没有塞进主仓库，可用 [`fp4/fetch_dependencies.ps1`](../../fetch_dependencies.ps1) 拉取固定 commit。

[`fp4/reference/`](../../reference/) 另有一份张量级实现，包含 MXFP4/6/8 网格、阈值公式、activation channel 排序、离线混合精度分区和 fake quant。它用于检查逻辑，不拿 CPU fake quant 时间代替官方 kernel 速度。
