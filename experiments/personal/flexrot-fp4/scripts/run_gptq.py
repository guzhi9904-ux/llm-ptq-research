from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    # vanilla GPTQ 与 MR-GPTQ 共用逐层实现；此入口只冻结关闭 ActOrder/MSE-grid。
    command = [sys.executable, str(__file__).replace("run_gptq.py", "run_mrgptq.py"), "--pipeline", "gptq", *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))
