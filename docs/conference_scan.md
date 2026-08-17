# 三大会论文与代码扫描

扫描范围默认指 ICLR、ICML 和 NeurIPS。MLSys、ACL、EMNLP 等会议中的相关方法作为扩展项，不与三大会主清单混写。

更新时间：2026-08-17。

## 已确认的主清单

| 年份 | 会议 | 方法 | 主要方向 | 官方代码 | 本地状态 |
|---:|---|---|---|---|---|
| 2023 | ICLR | GPTQ | 二阶近似、逐列误差补偿、weight-only | [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) | 已固定官方版本 |
| 2023 | ICML | SmoothQuant | 等价缩放、W8A8 | [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant) | 干净官方版本 |
| 2024 | ICLR | OmniQuant | LET、LWC、层级重构 | [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant) | 本地有修改 |
| 2024 | ICLR | AffineQuant | 可学习等价仿射变换 | [bytedance/AffineQuant](https://github.com/bytedance/AffineQuant) | 已重新取得官方版本 |
| 2024 | NeurIPS | QuaRot | 全局与在线 Hadamard 旋转、W4A4KV4 | [spcl/QuaRot](https://github.com/spcl/QuaRot) | 干净官方版本 |
| 2024 | NeurIPS Oral | DuQuant | 旋转与置换双变换、W4A4 | [Hsu1023/DuQuant](https://github.com/Hsu1023/DuQuant) | 干净官方版本 |
| 2025 | ICLR | SpinQuant | Cayley 优化的可学习旋转 | [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant) | 干净；非商业许可证 |
| 2025 | ICLR | OSTQuant | 可学习正交与缩放变换、QSUR、KL-Top | [BrotherHappy/OSTQuant](https://github.com/BrotherHappy/OSTQuant) | 已固定官方版本 |
| 2025 | ICML | FlatQuant | 每层可学习仿射变换、W4A4KV4 | [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant) | 本地有修改 |
| 2025 | ICML | GuidedQuant | 用最终损失梯度指导层级 PTQ | [snu-mllab/GuidedQuant](https://github.com/snu-mllab/GuidedQuant) | 已固定官方版本；候选增强方法 |
| 2025 | NeurIPS | Q-Palette | 旋转后的 weight-only 分数 bit 量化器 | [snu-mllab/Q-Palette](https://github.com/snu-mllab/Q-Palette) | 已固定官方版本；候选扩展方法 |
| 2026 | ICLR | MR-GPTQ / FP-Quant | 面向 MXFP4/NVFP4 的旋转与 GPTQ | [IST-DASLab/FP-Quant](https://github.com/IST-DASLab/FP-Quant) | 本地有修改；许可证待确认 |
| 2026 | ICLR | MicroMix | MXFP4/MXFP6/MXFP8 混合精度与 kernel 共设计 | [lwy2020/MicroMix](https://github.com/lwy2020/MicroMix) | 已固定 `micromix` 分支；无根 LICENSE |
| 2026 | ICLR | SliderQuant | 跨层和层内滑动式可学习 PTQ | [deep-optimization/SliderQuant](https://github.com/deep-optimization/SliderQuant) | 本地有少量修改 |
| 2026 | ICML | MixFP4 | NVFP4 block 内自适应选择 FP4/INT4 表示 | 公开代码尚未定位 | 未下载；阻塞 |

## 预印本与扩展项

| 方法 | 当前状态 | 纳入原因 | 注意事项 |
|---|---|---|---|
| ParoQuant | arXiv:2511.10645 | 硬件友好的 pairwise Givens rotation 与 channel-wise scaling | 当前不能标成三大会录用论文；main 分支与论文复现分支需区分 |
| DuQuant++ | arXiv:2604.17789 | 把 DuQuant 的细粒度旋转扩展到 MXFP4 | 官方代码已固定；会议状态待确认 |
| GPTQ-Babai | ICLR 2026 | 从 Babai nearest-plane 角度解释并改进 GPTQ | 作为 GPTQ 理论与实现扩展，不替代原始 GPTQ |
| AWQ | MLSys 2024 Best Paper | 经典 W4 weight-only 强基线 | 不属于三大会，但本地已有干净代码 |
| Atom | MLSys 2024 | W4A4、KV cache、mixed precision 与 kernel 共设计 | 可作为 INT4 系统实现扩展项 |
| QServe | MLSys 2025 | W4A8KV4 算法与系统共设计 | 可作为真实推理和速度实验扩展项 |

## 会议归属注意事项

- ParoQuant 当前只按公开预印本记录。
- MR-GPTQ 是论文提出的方法名，官方仓库名为 FP-Quant；目录中会同时保留两者的对应关系。
- MixFP4 已出现在 ICML 2026 正式论文列表，但在没有定位到作者官方代码前，不创建“已复现”或“已有代码”的标记。
- DuQuant++ 的 arXiv 标识和官方仓库已经出现，但正式会议归属仍需从权威页面确认。
