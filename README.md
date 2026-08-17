# LLM PTQ 论文代码与实验仓库

本仓库用于整理大语言模型后训练量化（LLM PTQ）的代表性论文、官方实现、统一实验配置和本次研究代码。

当前状态：**第二阶段——官方源码固定与许可证核查**。论文代码已从固定 Git commit 导出到各方法的 `upstream/`；没有直接复制本地第三方工作树，因此本地实验修改不会被误标为论文官方代码。

## 范围

INT 部分按方法路线组织：

- `int/gptq/`：GPTQ 及其目标函数、误差补偿和后续增强。
- `int/smooth/`：SmoothQuant 及等价缩放路线。
- `int/rotation/`：旋转与更广义的等价变换路线，包括 QuaRot、SpinQuant、ParoQuant、OmniQuant、AffineQuant、FlatQuant、DuQuant、OSTQuant 等。这里的部分方法并非“纯旋转”，具体标签以 `docs/method_matrix.md` 为准。

FP4 部分分为：

- `fp4/baselines/`：RTN、GPTQ、Rotation 三类基础基线。
- `fp4/methods/`：MR-GPTQ、MicroMix、MixFP4、DuQuant++ 等格式专用方法。
- `fp4/formats/`：MXFP4、NVFP4、基础 E2M1/E1M2 编码、尺度层级与模拟/真实内核定义。

`experiments/` 只放本次研究自己的代码、配置、运行记录和结果，不与论文原始实现混放。

## 目录

```text
llm-ptq-research/
├── docs/                 # 分类、会议扫描、算法与实验设置核查
├── manifests/            # 论文与本地源码清单
├── int/
│   ├── gptq/
│   ├── smooth/
│   └── rotation/
├── fp4/
│   ├── baselines/
│   ├── formats/
│   └── methods/
├── common/               # 后续统一接口，不放论文原实现
├── configs/              # 公共实验配置
├── environments/         # 方法隔离环境
├── third_party/          # 经许可证核查后固定的论文源码
├── experiments/          # 本次研究代码与设置
└── results/              # 结构化结果；默认不提交大模型和大缓存
```

## 代码接入原则

每篇论文最终采用以下结构：

```text
METHOD/
├── README.md             # 论文、算法、官方命令和核查状态
├── upstream/             # 可合法纳入时的固定官方源码
├── patches/              # 对官方源码的显式补丁
├── configs/              # 论文默认配置与统一对照配置分开保存
└── adapter/              # 接入公共评测框架的薄适配层
```

严格区分三套设置：

1. `paper-reported`：论文正文或附录声明的设置。
2. `upstream-default`：官方仓库实际默认值。
3. `reproduction`：本仓库为复现实验固定的设置。

三者不一致时不覆盖原值，而是在方法卡中并列记录。

## 当前入口

- 方法和会议清单：[`docs/conference_scan.md`](docs/conference_scan.md)
- 算法与参数矩阵：[`docs/method_matrix.md`](docs/method_matrix.md)
- 本地源码盘点：[`docs/local_inventory.md`](docs/local_inventory.md)
- 核查协议：[`docs/verification_protocol.md`](docs/verification_protocol.md)
- 中文编写规范：[`docs/chinese_writing_guide.md`](docs/chinese_writing_guide.md)
- 官方源码固定清单：[`manifests/upstream_sources.csv`](manifests/upstream_sources.csv)
- 许可证核查：[`docs/license_audit.md`](docs/license_audit.md)
- 本地修改差异：[`docs/local_diff_audit.md`](docs/local_diff_audit.md)
- 机器可读方法清单：[`manifests/methods.csv`](manifests/methods.csv)

## 重要限制

- 本地 `OmniQuant/`、`FlatQuant/`、`FP-Quant/` 和 `SliderQuant/` 存在修改或未跟踪文件；对应 `upstream/` 均从干净 Git ref 导出，差异没有混入。
- SpinQuant 本地许可证为 CC BY-NC 4.0，不能按宽松开源许可证处理。
- FP-Quant 与 MicroMix 根目录未发现许可证文件；当前快照只用于本地研究，公开发布前需要获得授权或改成外部引用。
- QuaRot、FlatQuant 和 MicroMix 的大型 submodule 没有递归复制；`.gitmodules` 与精确 commit 单独保留。
