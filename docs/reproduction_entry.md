# 统一复现实验入口设计

## 目标

统一入口解决四个问题：服务器路径不进入实验 YAML、方法依赖不互相污染、官方命令可审计、每次运行可追溯。它不是新的量化框架，也不重写论文算法。

## 三层结构

1. `configs/resources.local.yaml`：本机模型、数据集、缓存和输出路径。
2. `experiments/adapters/methods.yaml`：方法工作目录、环境档案和官方命令前缀。
3. `experiments/configs/*.yaml`：完整算法参数、官方 CLI 参数和实验标识。

解析后得到不经过 shell 拼接的命令数组。这样可以保留论文参数名差异，同时降低路径空格、引号和任意 shell 展开造成的错误。

## 命令语义

| 子命令 | 是否运行论文代码 | 用途 |
|---|---:|---|
| `list` | 否 | 查看方法接入状态与可用阶段 |
| `show` | 否 | 展示解析后的工作目录、命令和环境变量 |
| `check` | 否 | 检查资源路径、入口文件和可执行程序 |
| `doctor` | 否 | 比较当前环境与官方关键依赖版本 |
| `run --yes` | 是 | 建立不可覆盖结果目录并启动官方入口 |

## 当前边界

- GuidedQuant 和 SliderQuant 仍为 `manual`。前者是多阶段流水线，后者会通过任务队列文件改变运行状态，需继续拆分后再开放统一运行。
- MixFP4 为 `blocked`，原因是尚未找到作者官方公开代码。
- MR-GPTQ/FP-Quant 与 MicroMix 为 `runnable_local_only`，因为固定源码根目录没有明确 LICENSE。
- SpinQuant 为非商业许可证，入口会显示警告。
- `show/check` 可在 Windows 本地使用；大多数论文脚本和 CUDA kernel 的正式运行环境仍是 Linux。
- 公共日志和元数据统一写入 `run_dir`；论文代码自行产生的 checkpoint 或矩阵暂时保持官方目录语义，避免未经验证地修改源码保存逻辑。
