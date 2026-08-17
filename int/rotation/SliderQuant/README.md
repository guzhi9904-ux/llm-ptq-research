# SliderQuant

## 方法定位

SliderQuant 是 ICLR 2026 的 learnable PTQ 方法。官方 `main@eed0b8542f208c0e1d629ee3dd36f69416bb2a2e` 已导入 `upstream/`，许可证为 Apache-2.0。

## 核心逻辑

1. 根据 shallow、middle、deep layer 的不同 quantization sensitivity 设计 inter-layer sliding windows。
2. 在当前 window 内按增量方式执行 intra-layer sliding quantization。
3. 通过跨层耦合避免所有层独立、等权优化的限制。
4. 支持 W4A4、W2A16、dense 与 MoE 模型，并建立在 OmniQuant/QuaRot 代码基础上。

## 核心实验

官方代码用目录中的 `config.yaml` 控制完整参数，再用 `task_list.conf` 指向结果目录：

```bash
# 示例配置目录；其中 config.yaml 必须保存模型、bit、window 和优化参数。
result_dir=configs/llama2-7b-w2a16

# 单机多卡任务设置。
GPU_NUM=1
port=29507
THRESHOLD=0.05
WAIT_MODE=true
WAIT_INTERVAL=60
```

```bash
# 优化/量化。
./auto_train_ddp.sh

# 评测固定配置或 checkpoint。
./auto_test_one.sh
```

## 参数核查重点

- 三类 inter-layer window 的边界和滑动策略。
- intra-layer 增量量化顺序。
- `THRESHOLD` 在任务调度还是优化停止中的真实作用。
- W4A4 与 W2A16 配置中的 calibration data、epoch、learning rate 和 loss。
- dense、MoE、math/code 任务是否使用同一评测协议。

## 本地差异

原本本地 checkout 只有未跟踪的 layer observation smoke 输出和脚本；它们没有进入 `upstream/`。
