# GuidedQuant

## 方法定位

GuidedQuant 是 ICML 2025 方法，通过最终任务损失的 gradient information 改进层级 PTQ 目标。官方 `main@62bf3b8d6b54693ecccca6c7bbb32b3fcaded953` 已导入 `upstream/`，根目录许可证为 MIT。

## 核心逻辑

1. 在 calibration data 上计算最终 loss 对中间 weight/activation 的 gradient 或 saliency。
2. 把 end-loss guidance 融入局部 quantization objective，而不是只最小化 layer output MSE。
3. 保留同一 output channel 内 weight 的相互依赖。
4. 可增强 weight-only scalar、weight-only vector 和 W/A quantization backend。
5. 论文同时提出 LNQ 非均匀 scalar quantizer，并证明其目标单调下降。

## 环境

官方使用 Python 3.11、CUDA 12.4、pip 25.1。Any-Precision-LLM kernel 需要单独安装。

## 核心实验一：SqueezeLLM/LNQ + GuidedQuant

```bash
# 先下载官方 tokenized calibration data。
bash scripts/download_calibration.sh

# 参数依次为模型、bit 数和 gradient averaging group 数。
bash scripts/run_sqllm.sh meta-llama/Llama-2-7b-hf 2 4
bash scripts/run_lnq.sh meta-llama/Llama-2-7b-hf 2 4
```

## 核心实验二：QTIP/SpinQuant backend

官方记录两套不同 calibration protocol：

- QTIP：RedPajama，`1024 × 4096`。
- SpinQuant：WikiText-2，`128 × 2048`。

两者不能放进同一结果表而不注明数据差异。QTIP 示例：

```bash
cd qtip
bash exps/lufree_noft_qtip.sh 1mad 7b 2
```

## 核心实验三：真实推理速度

```bash
# 全精度示例。
python inference_example.py

# 2 bit LNQ + GuidedQuant，使用 Any-Precision-LLM kernel。
python inference_example.py -q
```

速度结果需要记录 RTX 3090、kernel 版本、是否 fuse Q/K/V 和 Up/Gate、生成长度及 `torch.compile` 设置。

## 复现边界

- GuidedQuant 是可插入多个 backend 的 objective enhancement，不把不同 backend 的结果混成单一量化器。
- 仓库内包含 Any-Precision、QTIP、SpinQuant 等组件，各自许可证继续有效。
