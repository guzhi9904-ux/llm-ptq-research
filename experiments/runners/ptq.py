#!/usr/bin/env python3
"""LLM PTQ 统一复现实验入口。

该入口不改写论文源码，只把资源别名解析为实际路径，并在固定的 upstream
工作目录中调用原始命令。默认只展示或检查；真正执行必须显式使用 run --yes。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 仅在入口环境缺依赖时触发
    raise SystemExit(
        "缺少 PyYAML。请执行：pip install -r environments/runner-requirements.txt"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "experiments" / "adapters" / "methods.yaml"
DEFAULT_PROFILES = REPO_ROOT / "environments" / "profiles.yaml"
DEFAULT_RESOURCES = REPO_ROOT / "configs" / "resources.local.yaml"
TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")
UNRESOLVED_ENV_RE = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


class ConfigError(RuntimeError):
    """表示统一配置无法安全解析。"""


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML，并要求顶层对象为映射。"""

    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"YAML 顶层必须是映射：{path}")
    return data


def dotted_get(data: dict[str, Any], dotted: str) -> Any:
    """按 a.b.c 形式读取嵌套字段。"""

    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"模板引用不存在：{dotted}")
        current = current[part]
    return current


def render_text(text: str, context: dict[str, Any]) -> str:
    """展开环境变量和双花括号模板，最多迭代十轮。"""

    rendered = os.path.expandvars(text)
    for _ in range(10):
        previous = rendered

        def replace(match: re.Match[str]) -> str:
            value = dotted_get(context, match.group(1))
            if value is None:
                raise ConfigError(f"模板值为空：{match.group(1)}")
            return str(value)

        rendered = TEMPLATE_RE.sub(replace, rendered)
        rendered = os.path.expandvars(rendered)
        if rendered == previous:
            break
    return rendered


def render_value(value: Any, context: dict[str, Any]) -> Any:
    """递归展开字符串，同时保留数字和布尔类型。"""

    if isinstance(value, str):
        return render_text(value, context)
    if isinstance(value, list):
        return [render_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, context) for key, item in value.items()}
    return value


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"缺少映射字段：{key}")
    return value


def select_resource(
    resources: dict[str, Any], group: str, alias: str | None, context: dict[str, Any]
) -> dict[str, Any] | None:
    """按别名选择模型或数据集，并生成统一 locator 字段。"""

    if alias is None:
        return None
    entries = require_mapping(resources, group)
    if alias not in entries or not isinstance(entries[alias], dict):
        raise ConfigError(f"资源别名不存在：{group}.{alias}")
    selected = dict(entries[alias])
    selected["alias"] = alias
    selected = render_value(selected, context)
    locator = selected.get("path") or selected.get("argument") or selected.get("identifier")
    if not locator:
        raise ConfigError(f"资源没有 path、argument 或 identifier：{group}.{alias}")
    selected["locator"] = str(locator)
    return selected


def normalize_runtime(experiment: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(experiment.get("runtime") or {})
    gpus = runtime.get("gpus", [0])
    if isinstance(gpus, str):
        gpu_text = gpus
        gpu_count = len([item for item in gpus.split(",") if item.strip()])
    elif isinstance(gpus, list):
        gpu_text = ",".join(str(item) for item in gpus)
        gpu_count = len(gpus)
    else:
        raise ConfigError("runtime.gpus 必须是列表或逗号分隔字符串")
    runtime.setdefault("python", sys.executable)
    runtime.setdefault("processes", max(gpu_count, 1))
    runtime.setdefault("master_port", 29500)
    runtime.setdefault("seed", 0)
    runtime["cuda_visible_devices"] = gpu_text
    return runtime


def build_plan(
    config_path: Path, resources_path: Path, registry_path: Path
) -> dict[str, Any]:
    """把实验、资源和方法注册表合并成可执行计划。"""

    experiment_file = load_yaml(config_path)
    resources_file = load_yaml(resources_path)
    registry_file = load_yaml(registry_path)
    experiment = require_mapping(experiment_file, "experiment")
    method_id = str(experiment.get("method") or "")
    stage_id = str(experiment.get("stage") or "")
    if not method_id or not stage_id:
        raise ConfigError("experiment.method 和 experiment.stage 必须显式填写")

    methods = require_mapping(registry_file, "methods")
    method = methods.get(method_id)
    if not isinstance(method, dict):
        raise ConfigError(f"未知方法：{method_id}")
    stages = method.get("stages") or {}
    if stage_id not in stages or not isinstance(stages[stage_id], dict):
        status = method.get("status", "unknown")
        note = method.get("note_zh", "")
        raise ConfigError(f"方法 {method_id} 没有阶段 {stage_id}；状态={status}。{note}")
    stage = stages[stage_id]

    roots_raw = require_mapping(resources_file, "roots")
    roots = {key: os.path.expandvars(str(value)) for key, value in roots_raw.items()}
    runtime = normalize_runtime(experiment_file)
    output = dict(experiment_file.get("output") or {})
    run_id = str(experiment.get("id") or config_path.stem)
    output.setdefault("run_dir", "{{ roots.output_root }}/{{ experiment.id }}")

    base_context: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "roots": roots,
        "runtime": runtime,
        "experiment": {**experiment, "id": run_id},
        "parameters": dict(experiment_file.get("parameters") or {}),
    }
    resource_refs = dict(experiment_file.get("resources") or {})
    model = select_resource(resources_file, "models", resource_refs.get("model"), base_context)
    dataset = select_resource(
        resources_file, "datasets", resource_refs.get("dataset"), base_context
    )
    base_context["model"] = model or {}
    base_context["dataset"] = dataset or {}
    output = render_value(output, base_context)
    base_context["output"] = output

    working_dir = REPO_ROOT / str(method["working_dir"])
    prefix = render_value(stage.get("command") or [], base_context)
    arguments = render_value(experiment_file.get("arguments") or [], base_context)
    if not isinstance(prefix, list) or not prefix:
        raise ConfigError(f"方法阶段没有命令：{method_id}.{stage_id}")
    if not isinstance(arguments, list):
        raise ConfigError("arguments 必须是参数列表，不能写成 shell 字符串")
    command = [str(item) for item in [*prefix, *arguments]]

    cache_root = roots.get("cache_root", "")
    controlled_env = {
        "CUDA_VISIBLE_DEVICES": runtime["cuda_visible_devices"],
        "PYTHONHASHSEED": str(runtime["seed"]),
        # 跨 Windows/Linux 固定子进程日志编码，避免中文输出被错误解码。
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": f"{cache_root}/huggingface",
        "HF_DATASETS_CACHE": f"{cache_root}/huggingface/datasets",
        # 老论文代码仍读取 TRANSFORMERS_CACHE，因此暂时同时设置。
        "TRANSFORMERS_CACHE": f"{cache_root}/huggingface/transformers",
        "TORCH_HOME": f"{cache_root}/torch",
    }
    extra_env = experiment_file.get("environment") or {}
    if not isinstance(extra_env, dict):
        raise ConfigError("environment 必须是键值映射")
    controlled_env.update(
        {str(key): str(value) for key, value in render_value(extra_env, base_context).items()}
    )

    return {
        "version": 1,
        "config_path": str(config_path.resolve()),
        "resources_path": str(resources_path.resolve()),
        "experiment": base_context["experiment"],
        "method": {
            "id": method_id,
            "display_name": method.get("display_name", method_id),
            "family": method.get("family"),
            "environment": method.get("environment"),
            "source_policy": method.get("source_policy"),
            "status": method.get("status"),
        },
        "stage": {
            "id": stage_id,
            "description_zh": stage.get("description_zh", ""),
            "required_files": list(stage.get("required_files") or []),
        },
        "resources": {"roots": roots, "model": model, "dataset": dataset},
        "runtime": runtime,
        "parameters": base_context["parameters"],
        "output": output,
        "working_dir": str(working_dir.resolve()),
        "command": command,
        "environment": controlled_env,
    }


def has_unresolved(text: str) -> bool:
    return bool(TEMPLATE_RE.search(text) or UNRESOLVED_ENV_RE.search(text))


def preflight(plan: dict[str, Any], check_executable: bool = True) -> tuple[list[str], list[str]]:
    """执行不导入论文代码的只读预检。"""

    errors: list[str] = []
    warnings: list[str] = []
    workdir = Path(plan["working_dir"])
    if not workdir.is_dir():
        errors.append(f"工作目录不存在：{workdir}")
    for relative in plan["stage"]["required_files"]:
        target = workdir / relative
        if not target.is_file():
            errors.append(f"官方入口文件不存在：{target}")

    for token in plan["command"]:
        if has_unresolved(token):
            errors.append(f"命令仍有未解析变量：{token}")
    for key, value in plan["environment"].items():
        if has_unresolved(value):
            errors.append(f"环境变量 {key} 仍有未解析路径：{value}")
    for key, value in plan["resources"]["roots"].items():
        if has_unresolved(str(value)):
            errors.append(f"资源根目录 {key} 仍有未解析变量：{value}")
    run_dir_text = str(plan["output"].get("run_dir", ""))
    if not run_dir_text:
        errors.append("output.run_dir 不能为空")
    elif has_unresolved(run_dir_text):
        errors.append(f"结果目录仍有未解析变量：{run_dir_text}")

    for resource_name in ("model", "dataset"):
        resource = plan["resources"].get(resource_name)
        if not resource:
            continue
        if resource.get("kind") == "local":
            locator = str(resource["locator"])
            if has_unresolved(locator):
                errors.append(f"{resource_name} 本地路径仍有未解析变量：{locator}")
            elif not Path(locator).exists():
                errors.append(f"{resource_name} 本地路径不存在：{locator}")
        if resource.get("revision") is None:
            warnings.append(f"{resource_name} 没有固定 revision，正式结果不可判为完全可复现")

    executable = plan["command"][0]
    if check_executable and not Path(executable).is_file() and shutil.which(executable) is None:
        errors.append(f"当前环境找不到可执行程序：{executable}")

    policy = plan["method"]["source_policy"]
    if policy == "local_only":
        warnings.append("该源码没有明确根 LICENSE，只允许本地研究运行，不应随仓库发布")
    elif policy == "noncommercial":
        warnings.append("该方法使用非商业许可证，运行结果和派生物不得用于商业用途")
    return errors, warnings


def print_plan(plan: dict[str, Any]) -> None:
    """用稳定、可复制的形式展示执行计划。"""

    print(f"实验：{plan['experiment']['id']}")
    print(f"方法：{plan['method']['display_name']} / {plan['stage']['id']}")
    print(f"环境：{plan['method']['environment']}")
    print(f"工作目录：{plan['working_dir']}")
    print(f"命令：{shlex.join(plan['command'])}")
    print("受控环境变量：")
    for key in sorted(plan["environment"]):
        print(f"  {key}={plan['environment'][key]}")


def run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_metadata(plan: dict[str, Any], config_bytes: bytes) -> dict[str, Any]:
    """只记录复现需要的系统信息，不复制整个用户环境。"""

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": plan["experiment"]["id"],
        "method": plan["method"],
        "stage": plan["stage"]["id"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(run_git(["status", "--porcelain"])),
        "platform": platform.platform(),
        "python": sys.version,
        "command": plan["command"],
        "working_dir": plan["working_dir"],
        "controlled_environment": plan["environment"],
    }
    try:
        import torch  # type: ignore

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            metadata["gpu_names"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # 入口环境通常不安装 torch
        metadata["torch_probe"] = f"不可用：{type(exc).__name__}"
    return metadata


def execute(plan: dict[str, Any], config_path: Path) -> int:
    """执行计划并保存解析配置、元数据与合并日志。"""

    run_dir = Path(str(plan["output"]["run_dir"]))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ConfigError(f"结果目录非空，拒绝覆盖既有运行：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_plan.yaml"
    metadata_path = run_dir / "metadata.json"
    command_path = run_dir / "command.txt"
    log_path = run_dir / "run.log"

    resolved_path.write_text(
        yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    config_bytes = config_path.read_bytes()
    metadata_path.write_text(
        json.dumps(collect_metadata(plan, config_bytes), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command_path.write_text(shlex.join(plan["command"]) + "\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.update(plan["environment"])
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            plan["command"],
            cwd=plan["working_dir"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return process.wait()


def parse_version(version: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", version.split("+")[0])
    return tuple(int(item) for item in numbers[:4])


def version_matches(actual: str, expected: Any) -> bool:
    spec = str(expected)
    if spec.startswith(">="):
        return parse_version(actual) >= parse_version(spec[2:])
    return actual == spec or actual.startswith(spec + "+")


def doctor(method_id: str, registry_path: Path, profiles_path: Path) -> int:
    registry = load_yaml(registry_path)
    methods = require_mapping(registry, "methods")
    method = methods.get(method_id)
    if not isinstance(method, dict):
        raise ConfigError(f"未知方法：{method_id}")
    profile_id = method.get("environment")
    profiles = require_mapping(load_yaml(profiles_path), "profiles")
    profile = profiles.get(profile_id)
    if not isinstance(profile, dict):
        raise ConfigError(f"没有环境档案：{profile_id}")

    failures = 0
    print(f"方法：{method.get('display_name', method_id)}")
    print(f"环境档案：{profile_id}（{profile.get('status')}）")
    print(f"证据：{profile.get('source')}")
    expected_python = profile.get("python")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if expected_python:
        ok = actual_python.startswith(str(expected_python) + ".") or actual_python == str(expected_python)
        print(f"Python：实际 {actual_python} / 期望 {expected_python} / {'通过' if ok else '不匹配'}")
        failures += int(not ok)
    else:
        print(f"Python：实际 {actual_python} / 官方未固定")

    for package, expected in (profile.get("packages") or {}).items():
        try:
            actual = importlib.metadata.version(package)
            ok = version_matches(actual, expected)
            state = "通过" if ok else "不匹配"
        except importlib.metadata.PackageNotFoundError:
            actual = "未安装"
            state = "缺失"
            ok = False
        print(f"{package}：实际 {actual} / 期望 {expected} / {state}")
        failures += int(not ok)
    print(f"说明：{profile.get('note_zh', '')}")
    return 1 if failures else 0


def list_methods(registry_path: Path) -> int:
    methods = require_mapping(load_yaml(registry_path), "methods")
    for method_id, method in methods.items():
        stages = ",".join((method.get("stages") or {}).keys()) or "-"
        print(
            f"{method_id:16} {method.get('status', 'unknown'):20} "
            f"env={str(method.get('environment')):14} stages={stages}"
        )
    return 0


def add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="实验 YAML")
    parser.add_argument(
        "--resources", type=Path, default=DEFAULT_RESOURCES, help="本机资源 YAML"
    )
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY, help="方法适配注册表"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM PTQ 统一复现实验入口")
    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="列出方法、状态和阶段")
    list_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)

    show_parser = subparsers.add_parser("show", help="解析并展示命令，不执行")
    add_plan_arguments(show_parser)

    check_parser = subparsers.add_parser("check", help="检查路径、入口和可执行程序")
    add_plan_arguments(check_parser)

    run_parser = subparsers.add_parser("run", help="保存运行元数据并执行官方入口")
    add_plan_arguments(run_parser)
    run_parser.add_argument("--yes", action="store_true", help="确认真正启动实验")

    doctor_parser = subparsers.add_parser("doctor", help="核对当前方法环境关键版本")
    doctor_parser.add_argument("--method", required=True)
    doctor_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    doctor_parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "list":
            return list_methods(args.registry)
        if args.action == "doctor":
            return doctor(args.method, args.registry, args.profiles)

        plan = build_plan(args.config, args.resources, args.registry)
        print_plan(plan)
        # show 用于跨平台审阅命令，不要求当前机器已经安装论文环境。
        errors, warnings = preflight(plan, check_executable=args.action != "show")
        for warning in warnings:
            print(f"警告：{warning}", file=sys.stderr)
        if errors:
            for error in errors:
                print(f"错误：{error}", file=sys.stderr)
            return 2
        print("解析完成。" if args.action == "show" else "预检通过。")
        if args.action in {"show", "check"}:
            return 0
        if not args.yes:
            print("未执行：真正运行需要显式添加 --yes。", file=sys.stderr)
            return 2
        return execute(plan, args.config)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
