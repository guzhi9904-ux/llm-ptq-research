# 本地源码盘点

盘点日期：2026-08-17。路径相对于当前工作区 `E:/graduateStudent/LLM Compress/project`。

| 方法/项目 | 本地路径 | 来源与 commit | 工作树 | 许可证观察 | 当前处理 |
|---|---|---|---:|---|---|
| SmoothQuant | `smoothquant/` | `mit-han-lab/smoothquant@c61476d728e4` | clean | MIT | 可固定为官方参考 |
| QuaRot | `QuaRot/` | `spcl/QuaRot@5008669b08c1` | clean | Apache-2.0 | 可固定为官方参考 |
| SpinQuant | `SpinQuant/` | `facebookresearch/SpinQuant@8f47aa3f00e8` | clean | CC BY-NC 4.0 | 只在许可证允许范围内使用 |
| ParoQuant | `paroquant/` | `z-lab/paroquant@f74a96c306f7` | clean | MIT | main 为当前产品化代码；论文复现需核对 legacy 分支 |
| OmniQuant | `OmniQuant/` | `OpenGVLab/OmniQuant@3a7d4fbc1749` | dirty，28 项 | MIT | 本地含非上游 LensQ/Qwen 修改，不直接搬运 |
| AffineQuant | `AffineQuant-main/` | 无 Git 元数据快照 | unknown | 旧快照来源不足 | 新仓库已重新取得官方固定版本 |
| FlatQuant | `FlatQuant/` | `ruikangliu/FlatQuant@9d88ffcb7d2c` | dirty，35 项 | MIT | 本地含 Qwen/部署修改，先保存差异 |
| DuQuant | `DuQuant/` | `Hsu1023/DuQuant@d56cfc6fe97c` | clean | MIT | 可固定为官方参考 |
| OSTQuant | 原工作区未下载 | `BrotherHappy/OSTQuant@ab64362da147` | clean snapshot | Apache-2.0 | 已导入新仓库 |
| FP-Quant / MR-GPTQ | `FP-Quant/` | `IST-DASLab/FP-Quant@d2e3092f9682` | dirty，5 项 | 根目录未发现 LICENSE | 先核实授权与本地差异 |
| MicroMix | `MicroMix-main/` | 旧快照无 Git 元数据 | unknown | 根目录未发现 LICENSE | 已重新固定官方 `micromix@c57370bc38f9`；仅本地研究 |
| MixFP4 | 未下载 | 论文已确认 | — | 待核实 | 官方代码尚未定位 |
| SliderQuant | `SliderQuant/` | `deep-optimization/SliderQuant@eed0b8542f20` | dirty，2 项 | Apache-2.0 | 候选扩展方法，先核实差异 |
| AWQ | `llm-awq/` | `mit-han-lab/llm-awq@d6e797a42b9e` | clean | MIT | MLSys 2024 扩展基线，不是三大会核心清单 |
| FlexRot-FP4 | `flexrot-fp4/` | `guzhi9904-ux/flexrot-fp4@eb5bb80b9e25` | clean | MIT | 本次实验候选代码，放 `experiments/` 路线 |

新仓库另外从远端固定了 GPTQ、GuidedQuant、Q-Palette 和 DuQuant++。完整 commit 与许可证见 `manifests/upstream_sources.csv`。

## 已有可复用核查材料

工作区已经存在：

- `experiment_inventory.md`：本地代码、commit、量化语义和历史结果盘点。
- `quantization_landscape.md`：INT/MXFP4/NVFP4、旋转、SmoothQuant、GPTQ/MR-GPTQ 的机制记录。
- `phase4a_smoothquant_mechanism_report.md`。
- `phase4b1_omniquant_mechanism_report.md`。
- `phase4b2_flatquant_mechanism_report.md`。
- `phase5a_gptq_compensation_report.md`。
- `phase5b_mrgptq_decomposition_report.md`。
- `quantization_harness/`：已有 INT、MXFP4、NVFP4 参考量化器与统一诊断基础。

这些材料属于 `local-historical` 或本地受控证据，不自动等价于论文官方复现。
