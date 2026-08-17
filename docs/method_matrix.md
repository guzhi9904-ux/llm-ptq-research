# 方法逻辑与实验设置矩阵

本文件记录第一轮核查结果。精确值以后会下沉到各方法目录的 `README.md` 和配置文件中。

## INT 方法

| 方法 | 核心对象 | 主要变换或优化 | 常见主设置 | 校准与目标 | 当前核查结论 |
|---|---|---|---|---|---|
| GPTQ | weight | Hessian 近似、逐列量化、误差反馈 | W4A16、W3A16 | 逐层输入激活；二阶加权重构 | 原始代码待固定；FP-Quant/QuaRot/SpinQuant/FlatQuant 各自带有衍生 GPTQ，不能混作同一实现 |
| SmoothQuant | weight + activation | 对角等价缩放，将 activation outlier 难度迁移到 weight | W8A8 | 512 条 Pile validation 句子收集 activation channel max | 官方代码是 W8A8 专用语义，不应直接当通用 W4A4 框架 |
| OmniQuant | weight 或 weight + activation | LET + LWC | W2/3/4A16、W6A6、W4A4 | 默认 128×2048、seed 2；block 输出 MSE | 默认量化器通常为非对称 affine；本地分支含非上游修改 |
| AffineQuant | weight 或 weight + activation | 可学习等价 affine matrix | W3A16、W4A4 | 默认 128 条校准样本、约 20 epoch | 需重新取得干净官方仓库；本地快照没有 Git 来源 |
| QuaRot | weight + activation + KV | 全局 residual rotation 与在线 Hadamard | W4A4KV4 | RTN 不需学习；GPTQ 使用校准激活 | fake-quant 主路径主要支持 LLaMA-2；支持 RTN/GPTQ |
| SpinQuant | weight + activation + KV | Cayley 参数化学习旋转矩阵 | W4A4KV4 | rotation optimization 100 steps；再运行 RTN/GPTQ | GPTQ W4A4 评测推荐使用 W16A4KV4 学到的 rotation，不能直接套 W4A4 训练命令 |
| DuQuant | weight + activation | block rotation + permutation | W4A4 | greedy rotation、permutation，可选 LWC | 本地 `run.sh` 使用 block 128、max step 256、alpha 0.6、LAC 0.9、SWC 0.8 |
| OSTQuant | weight + activation + KV | learnable orthogonal + scaling | W4A4KV4、W4-only | QSUR 指导，KL-Top 优化 | 官方代码与 ICLR 2025 已确认；待固定 commit 和真实默认值 |
| FlatQuant | weight + activation + KV | 每个 linear 的快速可学习 affine transform，可加 diagonal | W4A4KV4、W4A16 | 128 条校准样本；layer output MSE；15 epoch | activation per-token、weight per-channel；可选 GPTQ；本地树有修改 |
| ParoQuant | weight | independent pairwise Givens rotation + channel scale | INT4 weight-only | 学习 pairwise rotation | 当前 main 偏部署；论文复现需核对 legacy 分支 |

## FP4 基线与方法

| 方法 | 格式 | 核心逻辑 | 关键设置 | 当前核查结论 |
|---|---|---|---|---|
| RTN | MXFP4/NVFP4 | 按格式网格、block 和 scale 层级直接 round-to-nearest | MXFP4：block 32 + E8M0 scale；NVFP4：block 16 + E4M3 local scale + tensor scale | 两种格式不能只换 codebook；scale 规则是算法的一部分 |
| GPTQ | MXFP4/NVFP4 | 把 GPTQ 误差补偿应用到 FP4 网格 | FP-Quant 默认 update block 128、relative damping 0.01，可选 Static ActOrder | 必须说明 quantization group 与 GPTQ update block 的区别 |
| Rotation | MXFP4/NVFP4 | 权重离线旋转，activation 在线旋转 | Hadamard group 需要显式固定 | MXFP4 与 NVFP4 对旋转反应不同，不把 full Hadamard 当通用最优基线 |
| MR-GPTQ | MXFP4/NVFP4 | block-wise Hadamard + format-specific scale + GPTQ | 官方示例默认 Hadamard group 128、128×2048 FineWeb-Edu 校准 | 官方实现位于 FP-Quant；支持 LLaMA 与 Qwen3，不直接支持 Qwen2.5 |
| MicroMix | MXFP4/MXFP6/MXFP8 | 按 channel 分配混合 microscaling 精度并配套 GEMM kernel | 预处理默认 32×2048、`act_sort_metric=mean`、`lamda=1.0` | ICLR 2026；本地快照没有 Git 与许可证信息 |
| MixFP4 | NVFP4 派生 | 每个 block 自适应选择 E2M1 FP4 或 E1M2 INT4 风格表示 | 复用 NVFP4 scale hierarchy，借用 E4M3 block scale 的 sign bit 编码格式选择 | ICML 2026；代码未定位，当前只做论文级记录 |
| DuQuant++ | MXFP4 | 面向 microscaling block 的细粒度 rotation | 具体默认参数待官方代码核查 | 预印本与官方仓库已定位，会议归属待确认 |

## 统一对照时禁止混用的设置

以下差异会显著改变结果，必须作为显式配置维度：

- INT4 的 `[-7, 7]` 对称网格与 `[-8, 7]`/`2*amax/15` 网格。
- weight group size、activation group size、microscaling block size 和 GPTQ update block。
- activation 的 dynamic per-token scale 与校准得到的 static scale。
- MXFP4 的 E8M0 power-of-two scale 与 NVFP4 的两级 scale。
- max/absmax scale、MSE-grid scale 和 learnable clipping。
- RTN、GPTQ、Static ActOrder、true-sequential 是否开启。
- global、block-local、online、offline、random、learned rotation。
- fake quant PPL 与真实低比特 kernel 的速度/显存结果。
- WikiText-2 的 train/test split、512/2048 序列长度和滑窗策略。
