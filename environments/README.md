# 环境定义

论文仓库依赖的 PyTorch、Transformers、CUDA 和 kernel 版本存在实质冲突，因此本仓库统一调用方式，但不强行合并方法环境。

## 统一入口环境

统一入口只依赖 Python 与 PyYAML，不导入任何量化方法：

```bash
python -m venv .runner-venv
source .runner-venv/bin/activate
pip install -r environments/runner-requirements.txt
```

Windows PowerShell 的激活命令为：

```powershell
.runner-venv\Scripts\Activate.ps1
pip install -r environments/runner-requirements.txt
```

## 方法环境原则

1. 每个方法在独立 Conda/Mamba/venv 环境中安装。
2. 优先执行 `profiles.yaml` 中记录的官方安装入口。
3. 没有官方 lock 的方法不伪造精确版本，状态标为 `incomplete`。
4. CUDA kernel 编译环境与纯 fake-quant 精度环境分开记录。
5. 运行前使用 `doctor` 对当前 Python 和关键包版本做检查。

```bash
python experiments/runners/ptq.py doctor --method quarot
```

各环境的关键版本和证据来源见 `profiles.yaml`。完整依赖仍以固定 `upstream/` 中的文件为准，统一清单不替代官方文件。
