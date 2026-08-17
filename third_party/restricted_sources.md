# 无明确许可证源码的本地获取说明

FP-Quant 和 MicroMix 的作者官方仓库当前没有根目录 LICENSE。为了避免未经授权再分发，本仓库的 `.gitignore` 排除了它们的 `upstream/` 源码。

## FP-Quant / MR-GPTQ

- 官方仓库：`https://github.com/IST-DASLab/FP-Quant.git`
- 固定分支：`master`
- 固定 commit：`d2e3092f968262c4de5fb050e1aef568a280dadd`
- 本地目标：`fp4/methods/MR-GPTQ/upstream/`

研究者应直接从作者仓库取得该 commit，并自行遵守作者给出的使用条件。

## MicroMix

- 官方仓库：`https://github.com/lwy2020/MicroMix.git`
- 固定分支：`micromix`
- 固定 commit：`c57370bc38f999e1f75aada9c4b8a85baa44aaae`
- CUTLASS submodule commit：`a1aaf2300a8fc3a8106a05436e1a2abad0930443`
- 本地目标：`fp4/methods/MicroMix/upstream/`

MicroMix 的 ICLR 2026 分支不是仓库默认分支，获取时必须显式选择 `micromix`。
