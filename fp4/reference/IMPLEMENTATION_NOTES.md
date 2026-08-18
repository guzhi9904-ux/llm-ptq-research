# 代码和论文公式对照

## 代码来源

代码按论文公式、OCP microscaling 格式和 GPTQ 的基本做法编写。没有从本地 `fp4/methods/MR-GPTQ/upstream/` 或 `fp4/methods/MicroMix/upstream/` 复制函数、类、kernel 或脚本。

这份代码和论文仓库快照分别管理，不能看成作者官方版本。

## FP4 baseline 对应

| 论文里的做法 | 代码位置 | 固定参数 |
|---|---|---|
| absmax RTN | `baselines.py::FP4BaselineConfig` | `scale_strategy=minmax` |
| block Hadamard + RTN | 同上 | Hadamard group 默认等于格式 block size |
| 普通 GPTQ | 同上 | identity、default order、`relative_damp=1e-2` |

## MR-GPTQ 对应

| 论文里的做法 | 代码位置 | 怎么检查 |
|---|---|---|
| E2M1 元素网格 | `formats.py::positive_levels` | 精确端点测试 |
| MSE-optimized grid | `formats.py::_mse_scales` | 分组重构误差搜索 |
| MXFP scale fitting | `formats.py::fit_e8m0_scales` | 最多 256 个 log scale level |
| block Hadamard | `hadamard.py::block_hadamard` | 自逆与 Linear 等价测试 |
| static ActOrder | `mr_gptq.py::_gptq_with_fixed_scales` | 固定 scale 后按 Hessian 对角线排序 |
| GPTQ compensation | 同上 | inverse Hessian Cholesky 逐列更新 |

## MicroMix 对应

| 论文里的做法 | 代码位置 | 怎么检查 |
|---|---|---|
| 32 元素 MX block | `formats.py::microscale_fake_quant` | padding 恢复与 finite 测试 |
| 公式 6/21 阈值 | `micromix.py::quantization_threshold` | T6 大于 T4 测试 |
| 公式 9 channel 统计 | `micromix.py::channel_absolute_mean` | 升序 permutation 测试 |
| 公式 10 连续分区 | `micromix.py::build_micromix_plan` | 4/6/8-bit 覆盖全部 channel |
| 对应 weight/activation 精度 | `micromix.py::micromix_fake_linear` | 输出形状和有限值测试 |

## MixFP4 对应

| 论文里的做法 | 代码位置 | 怎么检查 |
|---|---|---|
| E1M2 正数网格 | `formats.py::positive_levels` | 精确比较 `0, 0.5, ..., 3.5` |
| tensor scale `max(abs(X))/2688` | `mixfp4.py::mixfp4_fake_quant` | 零张量和有限值测试 |
| E2M1/E1M2 两条 block scale | 同上 | 分别用最大幅值 6 和 7 |
| 逐 16 元素 block MSE 选择 | 同上 | 选择结果等于两条误差的较小值 |
| E1M2 到 INT4 风格网格 | `mixfp4.py::_quantize_e1m2_as_int4` | 按论文式 38 乘 2 |

## 还差哪些实验

接下来还要补模型层输入捕获、固定数据 revision、真实 tokenizer、逐层内存管理、PPL/lm-eval、packed weight 解码检查，以及 B200/RTX 5090 上的 kernel 对照。做完这些以后，才能和论文表格里的结果正式比较。
