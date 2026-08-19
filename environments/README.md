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

也可以直接让脚本创建隔离环境：

```powershell
# Windows：只安装 runner 依赖
powershell -ExecutionPolicy Bypass -File environments/bootstrap.ps1 -Profile runner

# Windows：安装 FP4 张量测试需要的固定依赖
powershell -ExecutionPolicy Bypass -File environments/bootstrap.ps1 -Profile fp4-reference
```

```bash
# Linux；第二个参数可以指定虚拟环境目录
bash environments/bootstrap.sh runner .runner-venv
bash environments/bootstrap.sh fp4-reference .fp4-reference-venv
```

固定文件的用途：

- `runner-requirements.txt`：只运行统一入口。
- `ci-requirements.txt`：CI 的 YAML 和 pytest 检查。
- `fp4-reference-requirements.txt`：CPU 上运行 FP4 合成张量测试。

PyTorch 的 CUDA wheel 必须按目标机器重新选择，不能拿 CPU 验证环境直接跑 kernel。

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

FlexRot-FP4 是仓库自己的实验模块，建议单独建立环境：

```bash
conda env create -f experiments/personal/flexrot-fp4/environment.yml
conda activate flexrot-fp4
pip install -e experiments/personal/flexrot-fp4
```

CI 只补充 `datasets` 以运行数据采样单元测试，不安装 `transformers` 和 `lm-eval`，也不会下载模型或数据集。

## 仓库级轻量验证

根目录的 `pyproject.toml` 已设置 FP4 参考包搜索路径。安装测试依赖后可以直接运行：

```bash
python tools/validate_repository.py
python -m pytest -q
```

GitHub Actions 也只执行这两类 CPU 检查，不下载模型、不编译 CUDA kernel、不把它当成论文结果复现。
