# 论文代码核查协议

## 状态标签

每个方法使用以下状态之一：

- `identified`：已找到论文和候选官方仓库。
- `source-pinned`：已固定官方仓库、分支、commit 和许可证。
- `logic-audited`：已沿入口、模型变换、量化器、校准和评测调用链核查。
- `config-audited`：已对照论文设置、仓库默认值和脚本命令。
- `smoke-tested`：最小模型或合成张量测试通过。
- `reproduced`：论文核心结果在允许误差内复现。
- `blocked`：缺代码、缺许可证、缺模型/硬件或设置不完整。

`README` 中的性能声明不能替代代码核查或本地复现。

## 1. 来源核查

每篇论文记录：

- 正式论文标题、作者、会议和年份；预印本不得标成已录用。
- arXiv/OpenReview/PMLR/NeurIPS Proceedings 链接。
- 作者或机构维护的官方仓库 URL。
- 固定 commit、分支、子模块 commit 和抓取日期。
- LICENSE 文件、代码许可证与模型/数据许可证。
- 本地工作树是否干净；有本地修改时保存 diff 清单，不覆盖官方状态。

## 2. 算法逻辑核查

沿真实执行入口确认：

- 量化对象：weight、activation、KV cache、embedding、LM head。
- 数值格式：INT、FP、MX、NV；有效编码、舍入、饱和和特殊值。
- 粒度：per-tensor、per-token、per-channel、per-group、microscaling block。
- scale/zero-point：对称或非对称，是否量化 scale，scale 的层级和选择目标。
- 校准：数据集、split、样本数、序列长度、seed、缓存位置。
- 补偿：Hessian/Gram 构造、damping、block size、ActOrder、true-sequential。
- 变换：缩放、置换、Hadamard、Givens、全局正交、可学习仿射及融合位置。
- 优化：目标函数、epoch/step、学习率、batch size、初始化和停止条件。
- 推理：fake quant、权重打包、真实低比特 GEMM 和在线 activation transform。

## 3. 实验设置核查

每种方法至少建立以下三份配置：

- `paper.yaml`：论文表格对应设置。
- `upstream.yaml`：官方脚本默认设置。
- `reproduction.yaml`：本仓库实际运行设置。

配置必须显式包含：

- 模型完整名称和 revision。
- dtype、设备、CUDA、PyTorch、Transformers 和量化内核版本。
- 校准与评测数据的 split 隔离。
- `wbits/abits/kvbits`、group/block size、symmetric、zero point。
- rotation 类型、大小、随机 seed、是否学习、是否离线融合。
- RTN/GPTQ、damping、update block、ActOrder、scale search。
- PPL 的序列长度和窗口策略。
- lm-eval 版本、任务、shot 数和 prompt 模板。
- 速度测试的 batch、prefill/decode 长度、warmup、重复次数和 GPU。

## 4. 正确性门槛

迁入公共比较表之前需要通过：

1. 编码端点、零值、舍入中点、饱和边界测试。
2. 等价变换的 FP32 前后输出一致性测试。
3. scale 层级和 block/group 边界测试。
4. fake quant 与打包/内核解码的数值对齐测试。
5. 单层输出误差、整模型 PPL 和至少一个下游任务的可重复性测试。
6. 校准集与评测集无重叠检查。

## 5. 结果证据等级

- `paper`：只来自论文。
- `upstream`：来自官方仓库现成日志或模型卡。
- `local-historical`：来自本地旧实验，但协议可能不同。
- `controlled`：在本仓库统一协议下产生。
- `kernel-measured`：真实低比特 kernel 实测；不可用 fake quant 速度代替。
