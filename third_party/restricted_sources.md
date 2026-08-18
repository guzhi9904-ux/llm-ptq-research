# 无明确根许可证的 FP4 源码

FP-Quant 和 MicroMix 的作者官方仓库当前没有根目录 LICENSE。用户已确认本地副本可以上传，因此固定源码已经放入各自的 `upstream/`。这项确认不应被写成上游提供了 MIT、Apache 等通用许可证。

## FP-Quant / MR-GPTQ

- 官方仓库：`https://github.com/IST-DASLab/FP-Quant.git`
- 固定分支：`master`
- 固定 commit：`d2e3092f968262c4de5fb050e1aef568a280dadd`
- 本地目标：`fp4/methods/MR-GPTQ/upstream/`

该目录与 commit 中 45 个文件逐项一致，没有带入本地 5 项修改。

## MicroMix

- 官方仓库：`https://github.com/lwy2020/MicroMix.git`
- 固定分支：`micromix`
- 固定 commit：`c57370bc38f999e1f75aada9c4b8a85baa44aaae`
- CUTLASS submodule commit：`a1aaf2300a8fc3a8106a05436e1a2abad0930443`
- 本地目标：`fp4/methods/MicroMix/upstream/`

MicroMix 的 ICLR 2026 分支不是仓库默认分支，获取时必须显式选择 `micromix`。

## 本仓库自己的复现代码

`fp4/reference/` 放的是按论文公式写的 PyTorch fake-quant 代码，使用 MIT License。它与上面的官方快照分开，不能用它的许可证覆盖第三方 `upstream/`。
