# FlexRot-FP4 单 RTX 4090 服务器运行手册

假设 Linux、单 NVIDIA RTX 4090、24GB VRAM，仓库目标为 `https://github.com/guzhi9904-ux/flexrot-fp4.git`。

## 1. Clone

```bash
git clone https://github.com/guzhi9904-ux/flexrot-fp4.git
cd flexrot-fp4
```

## 2. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate flexrot-fp4
python -m pip install -e .
```

若服务器 CUDA 驱动不兼容 `pytorch-cuda=12.1`，先用 `nvidia-smi` 确认驱动，再按 PyTorch 官方安装矩阵替换 CUDA 小版本；不要同时混装多个 torch wheel。

## 3. 安装与核对依赖

```bash
python - <<'PY'
import torch, transformers, datasets, yaml
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
PY
```

## 4. 查看 GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 5. 设置 Hugging Face 镜像与登录

网络需要时可选：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Llama 需要先在 Hugging Face 页面接受许可，再登录：

```bash
huggingface-cli login
```

不要把 token 写进 YAML 或提交到 Git。

## 6. 配置路径并下载模型

```bash
cp configs/paths.example.yaml configs/paths.yaml
vim configs/paths.yaml
python scripts/download_models.py qwen25_7b qwen3_8b llama31_8b
```

`configs/paths.yaml` 已被 `.gitignore` 忽略。推荐模型目录分别为 `/data/models/qwen25_7b`、`qwen3_8b`、`llama31_8b`。

## 7. 准备 WikiText-2

```bash
python scripts/prepare_wikitext.py
```

输出在 `dataset_root/wikitext-2-raw-v1`；calibration 固定用 train，PPL 固定用 test。

## 8. 历史 calibration cache（只用于旧结果核查）

下面的 BF16 full-model cache 不再用于论文对齐 RTN/GPTQ 比较，只在核查 Phase 7 历史结果时使用：

```bash
python scripts/build_calibration_cache.py --model qwen25_7b --num-sequences 8 --seq-len 512
python scripts/build_calibration_cache.py --model qwen3_8b --num-sequences 8 --seq-len 512
python scripts/build_calibration_cache.py --model llama31_8b --num-sequences 8 --seq-len 512
```

缓存立即从 GPU 移到 CPU，并逐模块写到 `cache_root/<model>/`。不要把 cache 加入 Git。

## 9. Paper-aligned 环境更新

更新仓库后安装新增的 `lm-eval` 依赖：

```bash
git pull
python -m pip install -e .
```

FineWeb-Edu 由 `datasets` 流式读取；可继续使用服务器的 `HF_ENDPOINT`。无需预先下载完整 10B-token 数据集。

## 10. Paper-aligned sanity check

```bash
pytest
python scripts/run_paper_experiments.py \
  --profile pilot --stage baseline \
  --models llama31_8b --formats nvfp4 \
  --pipelines rtn --seeds 42 \
  --skip-openllm --lm-eval-limit 0.001 --dry-run
```

去掉 `--dry-run` 才会真正运行。首次代码连通测试可加 `--skip-openllm`；正式 baseline 不得加。

## 11. Llama 3.1-8B pilot baseline（128×2048 FineWeb）

先跑 0°/45°、RTN/GPTQ/MR-GPTQ、三个 seed，并同时计算 PPL 与四项论文任务：

```bash
mkdir -p results/logs
set -o pipefail
python scripts/run_paper_experiments.py \
  --profile pilot \
  --stage baseline \
  --models llama31_8b \
  2>&1 | tee results/logs/llama31_8b_paper_pilot_baseline.log
```

单独先跑 MXFP4 MR-GPTQ：

```bash
python scripts/run_paper_experiments.py \
  --profile pilot --stage baseline \
  --models llama31_8b --formats mxfp4 \
  --pipelines mrgptq
```

## 12. baseline 完成后扫描弱角度

```bash
python scripts/run_paper_experiments.py \
  --profile pilot --stage flexrot \
  --models llama31_8b
```

若任一选定 model/format/pipeline/seed 的 0° 或 45° 不完整，命令会拒绝启动弱角度扫描。
弱角度默认只跑 PPL；确定候选角度后加 `--eval-openllm`，为所有选定 seed 补齐四项任务。

## 13. 最终 1024-sequence 复现

```bash
python scripts/run_paper_experiments.py \
  --profile paper --stage baseline \
  --models llama31_8b
```

`paper` profile 使用 1024×2048 FineWeb 和五个 seed；预计耗时远高于 pilot。

跨 seed 汇总：

```bash
python scripts/summarize_paper_results.py
column -s, -t < results/paper_aligned_summary.csv | less -S
```

## 14. 历史 Phase 8 命令

```bash
python scripts/run_rtn.py --model qwen25_7b --format nvfp4 --theta-deg 5.625
python scripts/run_mrgptq.py --model qwen25_7b --format nvfp4 --theta-deg 5.625 --pipeline mrgptq --resume
```

完整 Phase 8 会覆盖 YAML 中所有冻结角度。

## 15. 跑 Qwen3（历史 Phase 8）

```bash
python scripts/run_phase8.py --models qwen3_8b --pipelines rtn mrgptq
```

若 8B 在最终整模 PPL 时 OOM，改用已预注册回退：

```bash
python scripts/download_models.py qwen3_4b
python scripts/build_calibration_cache.py --model qwen3_4b
python scripts/run_phase8.py --models qwen3_4b --pipelines rtn mrgptq
```

必须在结果报告中标明发生了 8B→4B 回退。

## 16. 跑 Llama 3.1（历史 Phase 8）

```bash
python scripts/run_phase8.py --models llama31_8b --pipelines rtn mrgptq
```

出现 401/403 时检查是否在模型页面接受 Llama 3.1 许可、当前 shell 是否已登录以及镜像是否支持 gated repo。

## 17. 断点续跑

MR-GPTQ 每层写入 `modules/`、`sequential_state.pt` 和 `progress.json`：

```bash
python scripts/run_mrgptq.py --model qwen25_7b --format nvfp4 --theta-deg 5.625 --pipeline mrgptq --resume
```

不要手工混用不同 model/format/theta 的 output directory。RTN 若结果 JSON 状态为 `complete` 会自动跳过；需要重跑时显式加 `--force`。

## 18. 查看和汇总结果

```bash
python scripts/summarize_results.py
column -s, -t < results/summary.csv | less -S
```

历史 8×512 结果应检查 `predicted_tokens=4088`；paper-aligned 结果还必须检查 calibration dataset/长度/数量/seed、MXFP scale mode 和 OpenLLM 四项任务。

## 19. OOM 时怎么办

按以下顺序处理，并记录任何协议变化：

1. 确认没有第二个 Python 进程占 GPU：`nvidia-smi`。
2. 保持 layer-wise MR-GPTQ；不要把全部 Hessian 或 activation cache 搬到 GPU。
3. 降低 calibration forward 的同时窗口数量会改变协议，不应悄悄操作；优先减小内部 chunk，而不是减少总 token。
4. 关闭无关进程，重新从 `--resume` 启动；代码在每个 Linear 和 layer 后主动 `del`、`gc.collect()`、`torch.cuda.empty_cache()`。
5. Qwen3 允许按预注册规则回退 4B；Qwen2.5-7B 和 Llama-8B 不允许不记录地换模型。
6. 最终整模 PPL 仍 OOM 时，可把评测改成 CPU/offload 作为 technical debt 处理；不要把不同执行路径的结果混成同一表格。

## 历史 Phase 8 一键命令

```bash
python scripts/run_phase8.py
```

可先分模型运行降低故障半径：

```bash
python scripts/run_phase8.py --models qwen25_7b
python scripts/run_phase8.py --models qwen3_8b
python scripts/run_phase8.py --models llama31_8b
```
