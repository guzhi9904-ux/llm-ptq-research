# FlexRot-FP4

面向 FP4 大模型后训练量化的可调正交旋转强度方法。

本目录是主仓库中的个人实验模块。主仓库统一入口见 [`../../README.md`](../../README.md)，固定来源和接入差异见 [`SOURCE.md`](SOURCE.md)；下面保留独立运行方式，方便单独调试。

## 项目是什么

FlexRot-FP4 研究一个很具体的问题：在 W4A4 后训练量化中，正交旋转是否必须固定为 full Hadamard，还是应把旋转强度作为独立设计维度。项目提供可调的
`F(theta)^tensor(log2(block_size))` 旋转、NVFP4/MXFP4 fake quant、RTN、GPTQ、MR-GPTQ、WikiText-2 PPL 与逐层单 GPU pipeline。

## 核心动机

标准 Hadamard 固定为 45° 等幅 mixing，但不同 FP4 scale 体系对旋转后的分布并不等价：NVFP4 使用 block16 E4M3 local scale 与 FP32 global scale；MXFP4 使用 block32 E8M0 二次幂 scale。Phase 7 发现 NVFP4 更偏好较弱旋转，而 MXFP4 更偏好接近 full Hadamard 的强旋转，因此 rotation strength 不应被当作固定常数。

## 当前结果

Qwen2.5-1.5B-Instruct，WikiText-2，W4A4：

| Pipeline | 45° | FlexRot | 改善 |
|---|---:|---:|---:|
| NVFP4 RTN | 16.1237 | 15.3016（5.625°） | -0.8221 |
| NVFP4 GPTQ | 15.2078 | 15.1678（5.625°） | -0.0400 |
| NVFP4 MR-GPTQ | 15.0190 | 14.8483（5.625°） | -0.1708 |

MXFP4 MR-GPTQ 的反向控制为：33.75° = 16.5411，45° = 16.2788。它说明不同 FP4 format 的最优 strength 不同，而不是证明某个固定弱角度普遍最优。

## 项目结构

```text
configs/       模型、量化格式、Phase 7/8 与路径 YAML
flexrot/       rotation、FP4、GPTQ、calibration、evaluation、模型适配
scripts/       下载、缓存、RTN/GPTQ/MR-GPTQ、Phase 7/8 入口
tests/         数学端点、quantizer、GPTQ、配置与模块发现测试
docs/          来源追溯、协议、模型、环境与上传清单
results/       运行输出（除 .gitkeep 外默认忽略）
```

## 快速开始

```bash
conda env create -f environment.yml
conda activate flexrot-fp4
pip install -e .
cp configs/paths.example.yaml configs/paths.yaml
pytest
python scripts/prepare_wikitext.py
python scripts/download_models.py qwen25_15b
python scripts/run_paper_experiments.py --profile pilot --stage baseline --models llama31_8b --dry-run
```

论文对齐实验统一使用 FineWeb-Edu sequential calibration。pilot 为 128×2048、3 个 seed；最终 paper profile 为 1024×2048、5 个 seed。先完成 0°/45° baseline，之后 orchestrator 才允许弱角度扫描。

完整服务器步骤见 [README_服务器运行.md](README_服务器运行.md)，实验定义见 [README_实验说明.md](README_实验说明.md)。

## 当前支持模型

- Qwen2.5-1.5B-Instruct：Phase 7 已完整验证。
- Qwen2.5-7B-Instruct：Phase 8 同家族 scale-up。
- Qwen3-8B；单 4090 不足时允许回退 Qwen3-4B。
- Llama-3.1-8B-Instruct：跨模型族验证，Hugging Face gated。

## 当前实验阶段

- Phase 6：W16A4 activation-only，用来分离 activation 对旋转强度的响应。
- Phase 7：Qwen2.5-1.5B W4A4 angle landscape、GPTQ 与 MR-GPTQ 替换验证已完成。
- Phase 8：冻结 Phase 7 定性结论后做跨规模、跨代际和跨模型族验证。

## 重要限制

- 不宣称 5.625° 是 universal optimum。
- 当前仅在 Qwen2.5-1.5B-Instruct 上完成端到端验证。
- cross-model 结论必须等待 Phase 8 结果，不能由现有单模型结果外推。
- 本仓库实现的是可审计 fake quant/PTQ 研究路径，不是 RTX 4090 上的原生 FP4 Tensor Core kernel。
- RTN/GPTQ/MR-GPTQ 的结果可比较前提是模型、token 窗口、scale calibration、rotation support 和 predicted-token 规则全部一致。
