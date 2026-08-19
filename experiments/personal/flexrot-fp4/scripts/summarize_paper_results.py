from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import repository_paths, result_path


def _numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(payload.get("ppl"), (int, float)):
        metrics["wikitext2.ppl"] = float(payload["ppl"])
    for task, values in (payload.get("openllm_results") or {}).items():
        if not isinstance(values, dict):
            continue
        for metric, value in values.items():
            if isinstance(value, (int, float)) and "stderr" not in str(metric):
                metrics[f"{task}.{metric}"] = float(value)
    return metrics


def main() -> None:
    paths = repository_paths()
    root = result_path("paper_aligned", paths=paths)
    grouped: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for path in sorted(root.rglob("result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        relative = path.relative_to(root)
        profile = relative.parts[0] if relative.parts else "unknown"
        stage = relative.parts[1] if len(relative.parts) > 1 else "unknown"
        for metric, value in _numeric_metrics(payload).items():
            key = (
                profile,
                stage,
                payload.get("model"),
                payload.get("format"),
                payload.get("pipeline"),
                payload.get("theta_deg"),
                payload.get("calibration_sequences"),
                payload.get("calibration_seq_len"),
                metric,
            )
            grouped[key].append((int(payload["calibration_seed"]), value))

    rows = []
    for key, observations in sorted(grouped.items(), key=lambda item: str(item[0])):
        values = [value for _, value in observations]
        rows.append(
            {
                "profile": key[0],
                "stage": key[1],
                "model": key[2],
                "format": key[3],
                "pipeline": key[4],
                "theta_deg": key[5],
                "calibration_sequences": key[6],
                "calibration_seq_len": key[7],
                "metric": key[8],
                "n_seeds": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "seeds": " ".join(str(seed) for seed, _ in sorted(observations)),
            }
        )
    output = result_path("paper_aligned_summary.csv", paths=paths)
    fieldnames = [
        "profile", "stage", "model", "format", "pipeline", "theta_deg",
        "calibration_sequences", "calibration_seq_len", "metric", "n_seeds",
        "mean", "std", "min", "max", "seeds",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"汇总 {len(rows)} 个跨 seed 指标 -> {output}")


if __name__ == "__main__":
    main()
