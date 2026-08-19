"""统一入口的纯 CPU 测试，不导入任何论文实现。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "experiments" / "runners" / "ptq.py"
SPEC = importlib.util.spec_from_file_location("ptq_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RunnerTest(unittest.TestCase):
    def write_yaml(self, path: Path, data: dict) -> None:
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def make_files(self, root: Path, model_exists: bool = True) -> tuple[Path, Path]:
        model_path = root / "models" / "llama"
        if model_exists:
            model_path.mkdir(parents=True)
        output_path = root / "outputs"
        resources_path = root / "resources.yaml"
        config_path = root / "experiment.yaml"
        self.write_yaml(
            resources_path,
            {
                "version": 1,
                "roots": {
                    "model_root": str(root / "models"),
                    "dataset_root": str(root / "datasets"),
                    "cache_root": str(root / "cache"),
                    "output_root": str(output_path),
                },
                "models": {
                    "test_model": {
                        "kind": "local",
                        "path": "{{ roots.model_root }}/llama",
                        "revision": "test-snapshot",
                    }
                },
                "datasets": {
                    "c4": {
                        "kind": "builtin",
                        "argument": "c4",
                        "revision": "test-revision",
                    }
                },
            },
        )
        self.write_yaml(
            config_path,
            {
                "experiment": {
                    "id": "unit_gptq",
                    "method": "gptq",
                    "stage": "quantize_eval",
                },
                "resources": {"model": "test_model", "dataset": "c4"},
                "arguments": ["--wbits", "4", "--act-order"],
                "runtime": {"python": "python", "gpus": [2], "seed": 7},
                "output": {"run_dir": "{{ roots.output_root }}/unit_gptq"},
            },
        )
        return config_path, resources_path

    def test_build_plan_renders_paths_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path, resources_path = self.make_files(Path(temporary))
            plan = RUNNER.build_plan(config_path, resources_path, RUNNER.DEFAULT_REGISTRY)
            self.assertEqual(plan["method"]["id"], "gptq")
            self.assertEqual(plan["command"][-3:], ["--wbits", "4", "--act-order"])
            self.assertEqual(plan["environment"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertTrue(plan["resources"]["model"]["locator"].endswith("models/llama"))
            errors, warnings = RUNNER.preflight(plan, check_executable=False)
            self.assertEqual(errors, [])
            self.assertEqual(warnings, [])

    def test_preflight_rejects_missing_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path, resources_path = self.make_files(
                Path(temporary), model_exists=False
            )
            plan = RUNNER.build_plan(config_path, resources_path, RUNNER.DEFAULT_REGISTRY)
            errors, _ = RUNNER.preflight(plan, check_executable=False)
            self.assertTrue(any("model 本地路径不存在" in item for item in errors))

    def test_version_constraint(self) -> None:
        self.assertTrue(RUNNER.version_matches("2.2.1+cu121", "2.2.1"))
        self.assertTrue(RUNNER.version_matches("2.6.0", ">=2.0.0"))
        self.assertFalse(RUNNER.version_matches("2.1.0", "2.2.1"))

    def test_tracked_registry_entrypoints_exist(self) -> None:
        registry = RUNNER.load_yaml(RUNNER.DEFAULT_REGISTRY)
        for method_id, method in registry["methods"].items():
            if method["source_policy"] in {"local_only", "unavailable"}:
                continue
            workdir = REPO_ROOT / method["working_dir"]
            self.assertTrue(workdir.is_dir(), method_id)
            for stage_id, stage in method.get("stages", {}).items():
                for relative in stage.get("required_files", []):
                    self.assertTrue(
                        (workdir / relative).is_file(), f"{method_id}.{stage_id}: {relative}"
                    )

    def test_every_experiment_config_builds_and_passes_static_preflight(self) -> None:
        """不加载模型，只确认所有 YAML 能展开成入口和参数均存在的命令。"""

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            model_root = temporary_root / "models"
            dataset_root = temporary_root / "datasets"
            cache_root = temporary_root / "cache"
            output_root = temporary_root / "outputs"
            for relative in (
                "meta-llama/Llama-2-7b-hf",
                "meta-llama/Meta-Llama-3-8B",
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.1-8B",
                "meta-llama/Llama-3.2-1B-Instruct",
                "Qwen/Qwen3-8B",
            ):
                (model_root / relative).mkdir(parents=True)
            pile_path = dataset_root / "pile" / "val.jsonl.zst"
            pile_path.parent.mkdir(parents=True)
            pile_path.touch()
            cache_root.mkdir()
            output_root.mkdir()

            environment = {
                "PTQ_MODEL_ROOT": str(model_root),
                "PTQ_DATASET_ROOT": str(dataset_root),
                "PTQ_CACHE_ROOT": str(cache_root),
                "PTQ_OUTPUT_ROOT": str(output_root),
            }
            resources_path = REPO_ROOT / "configs" / "resources.example.yaml"
            config_paths = sorted((REPO_ROOT / "experiments" / "configs").glob("*.yaml"))
            with patch.dict(os.environ, environment, clear=False):
                for config_path in config_paths:
                    if config_path.name == "example_reproduction.yaml":
                        continue
                    with self.subTest(config=config_path.name):
                        plan = RUNNER.build_plan(
                            config_path, resources_path, RUNNER.DEFAULT_REGISTRY
                        )
                        errors, _ = RUNNER.preflight(plan, check_executable=False)
                        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
