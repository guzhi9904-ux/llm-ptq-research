from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="对单模型/格式运行固定角度 RTN 扫描")
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", choices=("nvfp4", "mxfp4"), required=True)
    parser.add_argument("--angles", nargs="+", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runner = Path(__file__).with_name("run_rtn.py")
    for angle in args.angles:
        command = [
            sys.executable,
            str(runner),
            "--model",
            args.model,
            "--format",
            args.format,
            "--theta-deg",
            str(angle),
            "--device",
            args.device,
        ]
        if args.force:
            command.append("--force")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
