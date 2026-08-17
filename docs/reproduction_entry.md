# 统一实验入口

## 这个入口做什么

这个脚本负责读取本机路径、拼好各方法的命令并保存运行记录。不同方法仍使用各自的 Python 环境，论文里的算法代码也不会在这里重写。

## 配置分成三部分

1. `configs/resources.local.yaml`：本机模型、数据集、缓存和输出路径。
2. `experiments/adapters/methods.yaml`：方法工作目录、环境档案和官方命令前缀。
3. `experiments/configs/*.yaml`：完整算法参数、官方 CLI 参数和实验标识。

脚本读完配置后直接生成命令参数列表，不经过 shell 字符串拼接。这样可以保留各仓库原来的参数名，也能少一些路径和引号问题。

## 可以使用的命令

| 子命令 | 是否运行论文代码 | 用途 |
|---|---:|---|
| `list` | 否 | 查看方法接入状态与可用阶段 |
| `show` | 否 | 展示解析后的工作目录、命令和环境变量 |
| `check` | 否 | 检查资源路径、入口文件和可执行程序 |
| `doctor` | 否 | 比较当前环境与官方关键依赖版本 |
| `run --yes` | 是 | 新建结果目录并启动官方入口 |

## 还没接好的部分

- GuidedQuant 和 SliderQuant 仍为 `manual`。前者是多阶段流水线，后者会通过任务队列文件改变运行状态，需继续拆分后再开放统一运行。
- MixFP4 为 `blocked`，原因是尚未找到作者官方公开代码。
- MR-GPTQ/FP-Quant 与 MicroMix 为 `runnable_local_only`，因为固定源码根目录没有明确 LICENSE。
- SpinQuant 为非商业许可证，入口会显示警告。
- `show/check` 可在 Windows 本地使用；大多数论文脚本和 CUDA kernel 的正式运行环境仍是 Linux。
- 公共日志和运行信息写入 `run_dir`。论文代码自己生成的 checkpoint 或矩阵暂时仍按原仓库的目录保存。
