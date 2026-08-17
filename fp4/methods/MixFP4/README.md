# MixFP4

## 方法定位

MixFP4 是 ICML 2026 论文《MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations》。当前已确认论文归属，但尚未定位作者官方代码，因此状态为 `blocked`。

## 论文级核心逻辑

1. 沿用 NVFP4 的两级 scale hierarchy。
2. 对每个 quantization block 在 E2M1 FP4 与 E1M2 INT4 风格表示之间自适应选择。
3. 复用 E4M3 block scale 的 sign bit 编码格式选择，不增加额外 metadata。
4. 目标是在保持 NVFP4 kernel 兼容性的同时，提高 outlier 或不同分布下的量化鲁棒性。

## 基线与核心消融计划

- NVFP4 RTN。
- 固定 E2M1。
- 固定 E1M2。
- oracle per-block E2M1/E1M2 选择。
- 论文 MixFP4 selector。
- selector metadata、面积、功耗和 tensor-core overhead。

## 暂不填写的内容

在找到作者官方代码前，不编造 selector 公式、阈值、校准样本数和运行命令。取得代码后，所有参数和核心实验均按中文注释规范补齐。
