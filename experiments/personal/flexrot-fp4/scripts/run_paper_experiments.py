from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import experiment_config, repository_paths, result_path


def _completed(path: Path) -> bool:
    result = path / "result.json"
    if not result.exists():
        return False
    return json.loads(result.read_text(encoding="utf-8")).get("status") == "complete"


def _job_dir(
    *,
    profile: str,
    stage: str,
    model: str,
    format_name: str,
    pipeline: str,
    seed: int,
    angle: float,
    paths,
) -> Path:
    return result_path(
        "paper_aligned",
        profile,
        stage,
        model,
        format_name,
        pipeline,
        f"seed_{seed}",
        f"theta_{angle:g}",
        paths=paths,
    )


def _require_baselines(
    *,
    profile: str,
    models: list[str],
    formats: list[str],
    pipelines: list[str],
    seeds: list[int],
    angles: list[float],
    paths,
) -> None:
    missing = []
    for model in models:
        for format_name in formats:
            for pipeline in pipelines:
                for seed in seeds:
                    for angle in angles:
                        path = _job_dir(
                            profile=profile,
                            stage="baseline",
                            model=model,
                            format_name=format_name,
                            pipeline=pipeline,
                            seed=seed,
                            angle=angle,
                            paths=paths,
                        )
                        if not _completed(path):
                            missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise SystemExit(
            f"FlexRot 扫描被阻止：缺少 {len(missing)} 个 paper baseline 结果。\n{preview}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 MR-GPTQ 论文基线或 FlexRot 扩展扫描")
    parser.add_argument("--profile", choices=("pilot", "paper"), default="pilot")
    parser.add_argument("--stage", choices=("baseline", "flexrot"), default="baseline")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--formats", nargs="+", choices=("nvfp4", "mxfp4"))
    parser.add_argument("--pipelines", nargs="+", choices=("rtn", "gptq", "mrgptq"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda")
    openllm_group = parser.add_mutually_exclusive_group()
    openllm_group.add_argument("--eval-openllm", action="store_true")
    openllm_group.add_argument("--skip-openllm", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-limit", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = experiment_config("paper_aligned")
    paths = repository_paths()
    models = args.models or list(config["models"])
    formats = args.formats or list(config["formats"])
    pipelines = args.pipelines or list(config["pipelines"])
    profile = config["calibration"]["profiles"][args.profile]
    seeds = args.seeds or [int(seed) for seed in profile["seeds"]]
    baseline_angles = [float(angle) for angle in config["baseline"]["angles"]]
    if args.stage == "flexrot":
        _require_baselines(
            profile=args.profile,
            models=models,
            formats=formats,
            pipelines=pipelines,
            seeds=seeds,
            angles=baseline_angles,
            paths=paths,
        )

    runner = Path(__file__).with_name("run_mrgptq.py")
    for model in models:
        for format_name in formats:
            angles = (
                baseline_angles
                if args.stage == "baseline"
                else [float(angle) for angle in config["flexrot"][f"{format_name}_angles"]]
            )
            for pipeline in pipelines:
                for seed in seeds:
                    for angle in angles:
                        output_dir = _job_dir(
                            profile=args.profile,
                            stage=args.stage,
                            model=model,
                            format_name=format_name,
                            pipeline=pipeline,
                            seed=seed,
                            angle=angle,
                            paths=paths,
                        )
                        command = [
                            sys.executable,
                            str(runner),
                            "--model", model,
                            "--format", format_name,
                            "--pipeline", pipeline,
                            "--theta-deg", str(angle),
                            "--device", args.device,
                            "--calibration-dataset", config["calibration"]["dataset"],
                            "--calibration-seq-len", str(config["calibration"]["seq_len"]),
                            "--num-calibration-sequences", str(profile["num_sequences"]),
                            "--calibration-seed", str(seed),
                            "--eval-seq-len", str(config["evaluation"]["ppl_seq_len"]),
                            "--num-eval-windows", str(config["evaluation"]["ppl_num_windows"]),
                            "--damping", str(config["quantization"]["damping"]),
                            "--traversal-chunk", str(config["quantization"]["traversal_chunk"]),
                            "--mxfp-scale-mode", config["quantization"]["mxfp_scale_mode"],
                            "--lm-eval-batch-size", args.lm_eval_batch_size,
                            "--output-dir", str(output_dir),
                            "--resume",
                        ]
                        openllm_default = config["evaluation"][f"{args.stage}_openllm"]
                        if (openllm_default or args.eval_openllm) and not args.skip_openllm:
                            command.append("--eval-openllm")
                        if args.skip_ppl:
                            command.append("--skip-ppl")
                        if args.lm_eval_limit is not None:
                            command.extend(("--lm-eval-limit", str(args.lm_eval_limit)))
                        print(" ".join(command), flush=True)
                        if not args.dry_run:
                            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
