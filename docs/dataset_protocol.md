# 数据集与校准协议

## 统一资源层与论文数据逻辑的边界

`configs/resources.local.yaml` 只负责把稳定别名映射到本地路径或 Hugging Face 标识符，不替换各论文的数据预处理代码。相同的 `c4` 名称在不同仓库中可能对应不同 tokenizer、采样窗口或预处理版本，因此结果中仍需记录方法入口和源码 commit。

## 当前公共别名

| 别名 | 用途 | 传给官方入口的值 | 关键要求 |
|---|---|---|---|
| `c4` | GPTQ、QuaRot 等校准 | `c4` | 固定数据 revision、采样 seed 和预处理版本 |
| `wikitext2` | PPL | `wikitext2` | 固定 raw 子集、test split 和 sequence length |
| `pile_validation_local` | SmoothQuant activation scale | 本地 `.jsonl.zst` 路径 | 官方核心设置为 512 条、长度 2048 |
| `fineweb_edu` | FP-Quant/MR-GPTQ 快速复现 | `fineweb-edu` | 官方 README 示例为 128 条、长度 2048；论文主实验为 1024 条 FineWeb |

## 数据隔离要求

1. 校准样本不得与最终 PPL 或任务评测样本重叠。
2. 每种方法记录 dataset identifier、subset、split、revision、seed、样本数和序列长度。
3. 使用本地快照时记录生成脚本、原始来源和文件校验和。
4. 数据下载与缓存目录不进入 Git；只提交 manifest 和校验信息。
5. lm-eval 任务必须同时记录 lm-eval 版本、任务名、shot 数和 batch size。

## 结果可比性

只有模型 revision、校准数据、序列长度、样本数、seed、量化网格和评测版本都一致时，才能把两个方法归为严格受控对照。仅数据集短名称相同不足以证明实验协议一致。
