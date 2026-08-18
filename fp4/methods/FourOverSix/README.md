# Four Over Six

Four Over Six 通过自适应 block scale 改进 NVFP4。直观上，它不再让所有 block 都机械地用最大幅值 6 对齐，而是在多个 scale rule 中选择误差更小的一条。作者仓库后来也加入了 IF4 等自适应 block-scaled data type。

## 源码

[`upstream/`](upstream/) 是 `mit-han-lab/fouroversix` commit `dadfad6901d473a734fe71e0b082e70ee993e23a` 的干净快照，包含：

- `src/fouroversix/quantize/pytorch/reference.py`：不依赖 Blackwell 的 PyTorch 参考实现。
- `src/fouroversix/quantize/triton_kernel.py`：Triton 量化 kernel。
- `src/fouroversix/csrc/`：CUDA/CUTLASS kernel。
- `scripts/ptq.py`：新 vLLM PTQ 入口。
- `scripts/ptq_hf/`：旧 Hugging Face PTQ 对照入口。
- `tests/`：格式和 kernel 测试。

仓库使用 MIT License。大型第三方依赖没有重复上传，可运行：

```powershell
powershell -ExecutionPolicy Bypass -File fp4/fetch_dependencies.ps1 -Method FourOverSix
```

## 基础命令

```bash
# 安装 PyTorch 参考实现；没有 Blackwell 时跳过 CUDA 编译
SKIP_CUDA_BUILD=1 pip install -e fp4/methods/FourOverSix/upstream

# 标准 NVFP4 RTN：固定 scale rule 6
python -m scripts.ptq_hf \
  --model-name meta-llama/Llama-3.2-1B \
  --ptq-method rtn \
  --task wikitext \
  --a-scale-rule static_6 \
  --w-scale-rule static_6

# Four Over Six RTN：使用方法默认的自适应 scale rule
python -m scripts.ptq_hf \
  --model-name meta-llama/Llama-3.2-1B \
  --ptq-method rtn \
  --task wikitext
```

真实测速要记录 GPU、CUDA、backend 和 shape。PyTorch reference 只用于核对数值，不用于速度结论。
