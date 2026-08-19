from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    # 正式 RTN 与 GPTQ/MR-GPTQ 共用同一个逐层校准与部署传播引擎。
    runner = Path(__file__).with_name("run_mrgptq.py")
    command = [sys.executable, str(runner), "--pipeline", "rtn", *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))
