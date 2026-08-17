# DuQuant++

## 方法定位

DuQuant++ 对应预印本《DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization》（arXiv:2604.17789），官方代码指向 [Hsu1023/DuQuant-v2](https://github.com/Hsu1023/DuQuant-v2)。

当前状态：官方 `main@6c13ec07a3241f2d4ad6415a06403aef43919db9` 已导入 `upstream/`；会议归属仍待确认。

## 核心方向

该方法把 DuQuant 的 outlier redistribution 思路扩展到 MXFP4，并把 rotation `block_size` 显式设为 32，与 MXFP4 原生 group 对齐；可选 GPTQ 补偿 rotation 后的 weight error。

## 核心实验计划

- MXFP4 RTN 与 DuQuant++ 对照。
- rotation 粒度消融。
- scale block 与 rotation block 对齐/不对齐消融。
- weight、activation 和 joint output error 分解。
- PPL、zero-shot 与真实 FP4 kernel 结果。

## 官方核心命令

```bash
# 不使用 GPTQ 的 DuQuant++ W4A4。
python main.py \
  --block_size 32 \
  --max_rotation_step 256 \
  --wbits 4 \
  --abits 4 \
  --model meta-llama/Llama-3-8B \
  --alpha 0.6 \
  --smooth \
  --eval_ppl \
  --batch_size 64 \
  --tasks arc_easy,arc_challenge,winogrande,hellaswag,openbookqa,lambada_openai,piqa
```

加入 `--gptq` 即为论文 README 中的 DuQuant++*。官方 README 写成了 `--bath_size 64`，但 `main.py` 的真实参数名是 `--batch_size`；本仓库使用可执行的正确参数名并保留这条差异记录。

## 参数核查

- `block_size`：argparse 默认 128，但论文核心命令显式使用 32；复现不能依赖默认值。
- `nsamples`：默认 128。
- `seed`：默认 2。
- `alpha`：argparse 默认 0.5，核心命令显式使用 0.6。
- activation：`per_token`；weight：`per_channel`。
- `max_rotation_step`：默认和核心命令均为 256。
