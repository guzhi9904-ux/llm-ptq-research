# 公共配置

这里保存跨方法共享的模型、数据集和路径定义。统一配置只负责资源定位，不覆盖论文代码中的算法默认值。

## 文件说明

- `resources.example.yaml`：可提交的资源模板，包含模型、数据集、缓存和结果根目录。
- `resources.local.yaml`：用户从模板复制得到的本机配置，已被 Git 忽略。
- 模型和数据集使用稳定别名；实验配置引用别名，不直接写服务器绝对路径。

## 初始化

```bash
cp configs/resources.example.yaml configs/resources.local.yaml
```

随后设置 `PTQ_MODEL_ROOT`、`PTQ_DATASET_ROOT`、`PTQ_CACHE_ROOT` 和 `PTQ_OUTPUT_ROOT`，或直接修改本机配置中的路径。

`kind: local` 的资源会在预检阶段验证路径是否存在；`kind: hub` 和 `kind: builtin` 只检查标识符，真正下载由对应论文环境负责。
