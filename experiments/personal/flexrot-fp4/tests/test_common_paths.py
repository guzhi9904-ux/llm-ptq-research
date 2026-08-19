from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_common_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "_common.py"
    spec = importlib.util.spec_from_file_location("flexrot_scripts_common", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_environment_paths_take_priority(monkeypatch, tmp_path: Path) -> None:
    common = _load_common_module()
    roots = {
        "model_root": tmp_path / "models",
        "dataset_root": tmp_path / "datasets",
        "cache_root": tmp_path / "cache",
        "result_root": tmp_path / "results",
    }
    for key, path in roots.items():
        monkeypatch.setenv(f"FLEXROT_{key.upper()}", str(path))

    explicit_model = tmp_path / "models" / "meta-llama" / "Llama-3.1-8B-Instruct"
    monkeypatch.setenv("FLEXROT_MODEL_PATH", str(explicit_model))

    assert common.repository_paths() == {key: path.resolve() for key, path in roots.items()}
    assert common.model_path("llama31_8b", roots) == explicit_model.resolve()
