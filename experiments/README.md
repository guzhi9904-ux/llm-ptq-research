# 本次实验代码与设置

本目录只放本次研究自己的实现、配置和结果，不复制论文原始代码。

## 计划结构

```text
experiments/
├── configs/          # 每个实验的完整 YAML，包含中文注释
├── runners/          # 统一运行入口，新增代码写中文模块说明
├── adapters/         # 对第三方论文代码的薄适配层
├── diagnostics/      # 层级误差、scale、code occupancy 等诊断
├── tests/            # 数值格式、等价变换和数据隔离测试
└── summaries/        # 中文实验总结
```

## 配置规则

- 每个结果必须能追溯到一个不可变配置文件。
- 模型 revision、代码 commit、数据 split、seed、dtype、硬件均显式记录。
- `paper`、`upstream` 和 `reproduction` 参数分开。
- 中文注释解释参数作用，参数名本身保持与代码一致。
- 默认不提交模型、缓存、checkpoint 和原始大日志。
