# LLM PTQ 论文代码与实验仓库

这个仓库用来集中整理 LLM PTQ 论文代码和自己的实验代码。

目前已经固定了各方法使用的 Git commit，并把模型、数据集和输出路径接入统一运行脚本。运行脚本只负责整理参数和调用原命令，不修改论文源码。

## 范围

INT 部分按方法路线组织：

- `int/gptq/`：GPTQ 及其目标函数、误差补偿和后续增强。
- `int/smooth/`：SmoothQuant 及等价缩放路线。
- `int/rotation/`：旋转与更广义的等价变换路线，包括 QuaRot、SpinQuant、ParoQuant、OmniQuant、AffineQuant、FlatQuant、DuQuant、OSTQuant 等。这里的部分方法并非“纯旋转”，具体标签以 `docs/method_matrix.md` 为准。

FP4 部分分为：

- `fp4/baselines/`：RTN、GPTQ、Rotation 三类基础基线。
- `fp4/methods/`：MR-GPTQ、MicroMix、MixFP4、FourOverSix、DuQuant++ 等格式专用方法。
- `fp4/formats/`：MXFP4、NVFP4、基础 E2M1/E1M2 编码、尺度层级与模拟/真实内核定义。
- `fp4/reference/`：自己编写的 FP4 基线、MR-GPTQ、MicroMix 和 MixFP4 参考代码。

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
├── configs/              # 模型、数据集、缓存和结果路径映射
├── environments/         # 方法隔离环境与关键版本档案
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
- 统一复现入口：[`docs/reproduction_entry.md`](docs/reproduction_entry.md)
- 数据集与校准协议：[`docs/dataset_protocol.md`](docs/dataset_protocol.md)
- 本机资源模板：[`configs/resources.example.yaml`](configs/resources.example.yaml)
- FP4 复现代码：[`fp4/reference/README.md`](fp4/reference/README.md)
- FP4 基线：[`fp4/baselines/README.md`](fp4/baselines/README.md)
- FP4 论文源码：[`fp4/methods/`](fp4/methods/)
- 个人实验代码：[`experiments/personal/flexrot-fp4/`](experiments/personal/flexrot-fp4/)
- 环境准备与轻量验证：[`environments/README.md`](environments/README.md)

## 统一入口快速示例

```bash
cp configs/resources.example.yaml configs/resources.local.yaml
python experiments/runners/ptq.py list
python experiments/runners/ptq.py show \
  --config experiments/configs/quarot_llama2_w4a4kv4.yaml \
  --resources configs/resources.local.yaml
```

环境只统一管理方式，不把依赖冲突的方法装进同一个 Python 环境。各方法的关键版本见 [`environments/profiles.yaml`](environments/profiles.yaml)。

不下载模型的仓库检查可以直接从根目录运行：

```bash
python tools/validate_repository.py
python -m pytest -q
```

GitHub Actions 使用 CPU 执行同一套检查，不运行论文模型和 CUDA kernel。

## 重要限制

- 本地 `OmniQuant/`、`FlatQuant/`、`FP-Quant/` 和 `SliderQuant/` 存在修改或未跟踪文件；对应 `upstream/` 均从干净 Git ref 导出，差异没有混入。
- SpinQuant 本地许可证为 CC BY-NC 4.0，不能按宽松开源许可证处理。
- FP-Quant 与 MicroMix 根目录未发现许可证文件。用户已确认本地副本可以上传；仓库仍保留缺少上游许可证的说明，避免被误当成 MIT/Apache 代码。
- QuaRot、FlatQuant、MicroMix 和 FourOverSix 的大型外部依赖没有重复上传；FP4 依赖可以用 `fp4/fetch_dependencies.ps1` 按固定 commit 拉取。
