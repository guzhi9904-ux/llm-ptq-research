#!/usr/bin/env python3
"""检查仓库自有配置、入口文件、中文 Markdown 链接和 Python 语法。"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "experiments" / "adapters" / "methods.yaml"
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
LOCAL_LINK_RE = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)#]+)(?:#[^)]+)?\)")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层不是映射：{path.relative_to(REPO_ROOT)}")
    return value


def validate_yaml_and_coverage(errors: list[str]) -> None:
    registry = load_yaml(REGISTRY_PATH)
    methods = registry.get("methods", {})
    covered: set[tuple[str, str]] = set()
    experiment_ids: set[str] = set()

    for config_path in sorted(CONFIG_DIR.glob("*.yaml")):
        if config_path.name == "example_reproduction.yaml":
            continue
        try:
            data = load_yaml(config_path)
            experiment = data["experiment"]
            method_id = str(experiment["method"])
            stage_id = str(experiment["stage"])
            experiment_id = str(experiment["id"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"配置字段错误：{config_path.name}: {exc}")
            continue
        if experiment_id in experiment_ids:
            errors.append(f"实验 id 重复：{experiment_id}")
        experiment_ids.add(experiment_id)
        method = methods.get(method_id)
        if not isinstance(method, dict):
            errors.append(f"配置引用未知方法：{config_path.name}: {method_id}")
            continue
        if stage_id not in (method.get("stages") or {}):
            errors.append(f"配置引用未知阶段：{config_path.name}: {method_id}.{stage_id}")
            continue
        covered.add((method_id, stage_id))

    for method_id, method in methods.items():
        workdir = REPO_ROOT / str(method.get("working_dir", ""))
        if not workdir.is_dir():
            errors.append(f"方法工作目录不存在：{method_id}: {workdir}")
        for stage_id, stage in (method.get("stages") or {}).items():
            if (method_id, stage_id) not in covered:
                errors.append(f"方法阶段没有统一配置：{method_id}.{stage_id}")
            for relative in stage.get("required_files", []):
                if not (workdir / relative).is_file():
                    errors.append(f"入口文件不存在：{method_id}.{stage_id}: {relative}")


def validate_markdown_links(errors: list[str]) -> None:
    for markdown_path in REPO_ROOT.rglob("*.md"):
        if "upstream" in markdown_path.parts or ".git" in markdown_path.parts:
            continue
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in LOCAL_LINK_RE.findall(text):
            target = raw_target.strip().replace("%20", " ")
            if not (markdown_path.parent / target).resolve().exists():
                relative = markdown_path.relative_to(REPO_ROOT)
                errors.append(f"Markdown 本地链接不存在：{relative}: {raw_target}")


def validate_owned_python(errors: list[str]) -> None:
    roots = [REPO_ROOT / "experiments", REPO_ROOT / "fp4" / "reference", REPO_ROOT / "tools"]
    for source_root in roots:
        for python_path in source_root.rglob("*.py"):
            try:
                ast.parse(python_path.read_text(encoding="utf-8"), filename=str(python_path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"Python 语法错误：{python_path.relative_to(REPO_ROOT)}: {exc}")


def main() -> int:
    errors: list[str] = []
    validate_yaml_and_coverage(errors)
    validate_markdown_links(errors)
    validate_owned_python(errors)
    if errors:
        for error in errors:
            print(f"错误：{error}", file=sys.stderr)
        return 1
    print("仓库轻量检查通过：配置覆盖、入口文件、Markdown 链接和自有 Python 语法正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
