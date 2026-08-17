# SpinQuant

## 方法定位

SpinQuant 是 ICLR 2025 的可学习旋转量化方法。官方代码为 [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant)，本地干净版本固定在 `8f47aa3f00e8662caf1a484153920a07e5281c3a`。

许可证为 CC BY-NC 4.0，不能按商业友好的宽松许可证处理。

## 核心逻辑

SpinQuant 保留 QuaRot 的函数等价旋转位置，但不把随机 Hadamard/正交旋转视为唯一选择。它使用 Cayley optimization 学习旋转矩阵，使量化后模型目标更优，再把学习到的旋转交给 RTN/GPTQ 评测流程。

## 核心实验一：学习 W4A4KV4 rotation

```bash
# 参数顺序依次为模型、weight bit、activation bit、KV bit。
bash scripts/10_optimize_rotation.sh \
  meta-llama/Llama-2-7b-hf 4 4 4
```

官方脚本中的关键值：

| 参数 | 官方脚本值 | 中文说明 |
|---|---:|---|
| `per_device_train_batch_size` | 1 | 每张 GPU 的 rotation 优化 batch |
| `model_max_length` | 2048 | 优化使用的序列长度 |
| `learning_rate` | 1.5 | Cayley rotation 优化学习率 |
| `max_steps` | 100 | 最大优化步数 |
| `bf16` | true | 优化计算 dtype |
| `k_groupsize/v_groupsize` | 128 | K/V cache 量化分组 |

## 核心实验二：用学习 rotation 做 PTQ

```bash
# 需要先把脚本中的 optimized_rotation_path 指向上一步生成的 R.bin。
bash scripts/2_eval_ptq.sh \
  meta-llama/Llama-2-7b-hf 4 4 4
```

评测脚本默认 `per_device_eval_batch_size=4`、`model_max_length=2048`、BF16，并开启 weight clipping、activation/K/V 非对称量化以及 K/V group size 128。

## GPTQ 特别注意

如果最终用 GPTQ 做 W4A4KV4，官方 README 推荐先在 `W16A4KV4` 网络上学习 rotation：

```bash
# 学 rotation 时不量化 weight，避免 rotation 优化与 GPTQ weight 更新重复耦合。
bash scripts/10_optimize_rotation.sh meta-llama/Llama-2-7b-hf 16 4 4

# 随后用上一步的 rotation 做 W4A4KV4 GPTQ 评测。
bash scripts/2_eval_ptq.sh meta-llama/Llama-2-7b-hf 4 4 4
```

## 复现边界

- 论文内部 LLaMA 代码与公开 Hugging Face 代码存在实现来源差异，结果需标注所用代码路径。
- 没有 `--w_rtn` 时评测路径使用 GPTQ；打开 `--w_rtn` 才是 RTN。
