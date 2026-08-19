from __future__ import annotations

from pathlib import Path

import yaml


def load_model_spec(name: str, *, config_root: str | Path = "configs/models") -> dict:
    """按短名称读取模型 YAML；模型 ID 与架构路径不硬编码在 pipeline 中。"""

    path = Path(config_root) / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"未知模型 {name!r}：未找到 {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("name") != name:
        raise ValueError(f"模型配置格式错误：{path}")
    return payload


def supported_models(*, config_root: str | Path = "configs/models") -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in Path(config_root).glob("*.yaml")))
