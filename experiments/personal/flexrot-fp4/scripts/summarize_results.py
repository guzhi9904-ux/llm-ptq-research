from __future__ import annotations

import csv
import json

from _common import repository_paths, result_path


def main() -> None:
    paths = repository_paths()
    root = paths["result_root"]
    rows = []
    candidates = set(root.rglob("result.json")) | set(root.rglob("theta_*.json"))
    for path in sorted(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete" or "ppl" not in payload:
            continue
        rows.append(
            {
                "model": payload.get("model"),
                "format": payload.get("format"),
                "pipeline": payload.get("pipeline"),
                "theta_deg": payload.get("theta_deg"),
                "ppl": payload.get("ppl"),
                "predicted_tokens": payload.get("predicted_tokens"),
                "path": str(path),
            }
        )
    output = result_path("summary.csv", paths=paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model", "format", "pipeline", "theta_deg", "ppl", "predicted_tokens", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"汇总 {len(rows)} 个完成结果 -> {output}")


if __name__ == "__main__":
    main()
