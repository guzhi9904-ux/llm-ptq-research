from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 跨模型一键运行入口")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--pipelines", nargs="*", choices=("rtn", "gptq", "mrgptq"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()
    config = experiment_config("phase8_cross_model")
    models = args.models or config["models"]
    pipelines = args.pipelines or list(config["pipelines"])
    if args.include_optional:
        pipelines += [name for name in config["optional_pipelines"] if name not in pipelines]
    scripts = {
        "rtn": Path(__file__).with_name("run_rtn.py"),
        "gptq": Path(__file__).with_name("run_gptq.py"),
        "mrgptq": Path(__file__).with_name("run_mrgptq.py"),
    }
    for model in models:
        for format_name, angles in (("nvfp4", config["nvfp4_angles"]), ("mxfp4", config["mxfp4_angles"])):
            for pipeline in pipelines:
                for angle in angles:
                    command = [
                        sys.executable,
                        str(scripts[pipeline]),
                        "--model",
                        model,
                        "--format",
                        format_name,
                        "--theta-deg",
                        str(angle),
                        "--device",
                        args.device,
                    ]
                    if pipeline != "rtn":
                        command.append("--resume")
                    print(" ".join(command), flush=True)
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
