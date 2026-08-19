from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 并要求顶层为 mapping，避免配置类型错误拖到长实验中才暴露。"""

    resolved = Path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 顶层必须为 mapping：{resolved}")
    return payload


def resolve_path(value: str, *, root: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(root) / path).resolve()


def load_paths(path: str | Path = "configs/paths.yaml") -> dict[str, Path]:
    payload = load_yaml(path)
    required = {"model_root", "dataset_root", "cache_root", "result_root"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"paths 配置缺少：{sorted(missing)}")
    base = Path(path).resolve().parents[1]
    return {key: resolve_path(str(payload[key]), root=base) for key in required}
