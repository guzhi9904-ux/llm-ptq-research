# 独立实现说明与公式对应

## 来源边界

本目录以论文公开公式、OCP microscaling 格式描述和经典 GPTQ 思路为规范重新设计代码结构。没有从本地 `fp4/methods/MR-GPTQ/upstream/` 或 `fp4/methods/MicroMix/upstream/` 复制函数、类、kernel 或脚本。

独立实现不改变两个官方仓库的许可证状态，也不代表原作者认可本实现。

## MR-GPTQ 对应

| 论文成分 | 本实现 | 核查方式 |
|---|---|---|
| E2M1 元素网格 | `formats.py::positive_levels` | 精确端点测试 |
| MSE-optimized grid | `formats.py::_mse_scales` | 分组重构误差搜索 |
| MXFP scale fitting | `formats.py::fit_e8m0_scales` | 最多 256 个 log scale level |
| block Hadamard | `hadamard.py::block_hadamard` | 自逆与 Linear 等价测试 |
| static ActOrder | `mr_gptq.py::_gptq_with_fixed_scales` | 固定 scale 后按 Hessian 对角线排序 |
| GPTQ compensation | 同上 | inverse Hessian Cholesky 逐列更新 |

## MicroMix 对应

| 论文成分 | 本实现 | 核查方式 |
|---|---|---|
| 32 元素 MX block | `formats.py::microscale_fake_quant` | padding 恢复与 finite 测试 |
| 公式 6/21 阈值 | `micromix.py::quantization_threshold` | T6 大于 T4 测试 |
| 公式 9 channel 统计 | `micromix.py::channel_absolute_mean` | 升序 permutation 测试 |
| 公式 10 连续分区 | `micromix.py::build_micromix_plan` | 4/6/8-bit 覆盖全部 channel |
| 对应 weight/activation 精度 | `micromix.py::micromix_fake_linear` | 输出形状和有限值测试 |

## 与正式复现的差距

下一步要达到 `reproduced`，仍需加入模型层捕获、固定数据 revision、真实 tokenizer、逐层内存管理、PPL/lm-eval、packed weight 解码对齐，以及 B200/RTX 5090 上的官方 kernel 对照。任何论文表格结果都必须保留官方实现与本参考实现两条独立证据链。
