from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import experiment_config, repository_paths, result_path, write_json


def _rtn_path(angle: float, paths) -> Path:
    return result_path("runs", "qwen25_15b", "nvfp4", "rtn", f"theta_{angle:g}.json", paths=paths)


def _mrgptq_path(angle: float, paths) -> Path:
    return result_path("runs", "qwen25_15b", "nvfp4", "mrgptq", f"theta_{angle:g}", "result.json", paths=paths)


def _verify(path: Path, expected: float, tolerance: float) -> dict:
    if not path.exists():
        return {"path": str(path), "status": "missing", "expected": expected}
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = float(payload["ppl"])
    relative = actual / expected - 1.0
    return {
        "path": str(path),
        "status": "passed" if abs(relative) <= tolerance else "failed",
        "expected": expected,
        "actual": actual,
        "relative_delta": relative,
        "predicted_tokens": payload.get("predicted_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行或检查 Phase 7 五个上传门禁 arm")
    parser.add_argument("--check-only", action="store_true", help="不启动实验，只检查新仓库已有输出")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = experiment_config("phase7_reproduction")
    paths = repository_paths()
    if not args.check_only:
        raise SystemExit(
            "Phase 7 的 BF16 cache RTN 路径已冻结为历史协议；"
            "只能用 --check-only 审核已有结果。新实验请运行 run_paper_experiments.py。"
        )

    tolerance = float(config["expected_results"]["tolerance_relative"])
    rows = []
    for angle_text, expected in config["expected_results"]["nvfp4_rtn"].items():
        rows.append(_verify(_rtn_path(float(angle_text), paths), float(expected), tolerance))
    for angle_text, expected in config["expected_results"]["nvfp4_mrgptq"].items():
        rows.append(_verify(_mrgptq_path(float(angle_text), paths), float(expected), tolerance))
    gate = {
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
        "tolerance_relative": tolerance,
        "rows": rows,
    }
    output = result_path("phase7_reproduction", "gate.json", paths=paths)
    write_json(output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if gate["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
