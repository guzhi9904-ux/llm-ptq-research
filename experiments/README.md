# 本次实验代码与设置

本目录保存本次研究自己的配置、薄适配层、运行入口和测试，不复制或修改论文原始实现。

## 目录

```text
experiments/
├── adapters/methods.yaml    # 方法工作目录、环境和官方入口前缀
├── configs/                 # 带中文注释的不可变实验 YAML
├── runners/ptq.py           # list/show/check/doctor/run 统一入口
├── tests/                   # 不需要 GPU 的配置与预检测试
├── diagnostics/             # 后续层级误差与量化格式诊断
└── summaries/               # 中文实验总结
```

## 最短使用流程

```bash
# 1. 准备本机模型、数据集、缓存和结果路径。
cp configs/resources.example.yaml configs/resources.local.yaml

# 2. 查看当前已经接入的方法和阶段。
python experiments/runners/ptq.py list

# 3. 只解析命令，不启动 GPU 实验。
python experiments/runners/ptq.py show \
  --config experiments/configs/quarot_llama2_w4a4kv4.yaml \
  --resources configs/resources.local.yaml

# 4. 检查本地路径、官方入口和可执行程序。
python experiments/runners/ptq.py check \
  --config experiments/configs/quarot_llama2_w4a4kv4.yaml \
  --resources configs/resources.local.yaml

# 5. 在 QuaRot 的隔离环境中确认关键版本。
python experiments/runners/ptq.py doctor --method quarot

# 6. 只有显式确认后才真正运行。
python experiments/runners/ptq.py run \
  --config experiments/configs/quarot_llama2_w4a4kv4.yaml \
  --resources configs/resources.local.yaml \
  --yes
```

## 配置规则

- `parameters` 是便于论文对照和结果分析的语义化记录。
- `arguments` 是实际传给官方代码的参数数组，两者都必须填写，不能只记录其中一份。
- 参数名保持官方实现原样；统一入口不把 `wbits`、`w_bits` 等强行改成同一个名字。
- 模型 revision、数据 revision、split、seed、序列长度和校准样本数必须显式记录。
- `paper-reported`、`upstream-default`、`reproduction` 和 `controlled` 不得混写。
- 默认不提交模型、缓存、checkpoint 和原始大日志。

## 运行产物

一次正式运行至少生成：

- `resolved_plan.yaml`：所有路径和模板解析后的完整计划。
- `metadata.json`：Git commit、系统、Python、PyTorch/CUDA 探测和受控环境变量。
- `command.txt`：不经过 shell 拼接的最终命令。
- `run.log`：stdout 与 stderr 合并日志。

结果目录非空时入口会拒绝覆盖，防止后一次实验污染前一次记录。

部分官方代码仍会把 checkpoint、矩阵或预处理张量写入它自己的相对目录。统一入口当前保证公共元数据和日志进入 `run_dir`，但不会通过修改官方源码强制搬运这些原生产物；正式运行后应在实验总结中登记原生产物路径。MicroMix 的 `saved/`、QuaRot 的 `experiments/` 等目录尤其需要检查。
