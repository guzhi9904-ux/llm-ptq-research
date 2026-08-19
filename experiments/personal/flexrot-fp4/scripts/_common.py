from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flexrot.utils.config import load_paths, load_yaml  # noqa: E402


def repository_paths() -> dict[str, Path]:
    # 统一实验入口通过环境变量传入四类根目录；独立运行时仍兼容原来的 paths.yaml。
    environment_keys = {
        "model_root": "FLEXROT_MODEL_ROOT",
        "dataset_root": "FLEXROT_DATASET_ROOT",
        "cache_root": "FLEXROT_CACHE_ROOT",
        "result_root": "FLEXROT_RESULT_ROOT",
    }
    environment_values = {
        key: os.environ.get(variable) for key, variable in environment_keys.items()
    }
    if any(environment_values.values()):
        missing = [
            variable
            for key, variable in environment_keys.items()
            if not environment_values[key]
        ]
        if missing:
            raise ValueError(f"FlexRot 路径环境变量没有填完整：{missing}")
        return {
            key: Path(str(value)).expanduser().resolve()
            for key, value in environment_values.items()
        }

    path = REPO_ROOT / "configs" / "paths.yaml"
    if not path.exists():
        raise FileNotFoundError(
            "缺少 configs/paths.yaml；请先复制 configs/paths.example.yaml 并修改路径"
        )
    return load_paths(path)


def model_config(name: str) -> dict:
    return load_yaml(REPO_ROOT / "configs" / "models" / f"{name}.yaml")


def experiment_config(name: str) -> dict:
    return load_yaml(REPO_ROOT / "configs" / "experiments" / f"{name}.yaml")


def model_path(name: str, paths: dict[str, Path]) -> Path:
    # 统一配置可以直接给出模型快照目录，避免依赖某一种 model_root 布局。
    explicit = os.environ.get("FLEXROT_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    direct = paths["model_root"] / name
    if direct.exists():
        return direct
    # 兼容从 HF snapshot 保留官方仓库目录名的本地资产布局。
    official_name = str(model_config(name)["huggingface_id"]).rsplit("/", 1)[-1]
    official = paths["model_root"] / official_name
    return official if official.exists() else direct


def dataset_path(paths: dict[str, Path]) -> Path:
    return paths["dataset_root"] / "wikitext-2-raw-v1"


def cache_path(name: str, paths: dict[str, Path]) -> Path:
    direct = paths["cache_root"] / name
    if direct.exists():
        return direct
    # 开发机可只读复用旧 cache；优先不带 seed 后缀的正式 BF16 cache。
    canonical = paths["cache_root"] / f"{name}_all_linears_4096train_1024val_bf16"
    return canonical if canonical.exists() else direct


def result_path(*parts: str, paths: dict[str, Path]) -> Path:
    return paths["result_root"].joinpath(*parts)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
