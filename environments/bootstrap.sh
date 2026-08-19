#!/usr/bin/env bash
set -euo pipefail

profile="${1:-runner}"
environment_path="${2:-.venv}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_path="${repo_root}/${environment_path}"

if [[ "${profile}" != "runner" && "${profile}" != "fp4-reference" ]]; then
  echo "profile 只能是 runner 或 fp4-reference" >&2
  exit 2
fi

python3 -m venv "${target_path}"
environment_python="${target_path}/bin/python"
"${environment_python}" -m pip install --upgrade pip
"${environment_python}" -m pip install -r "${repo_root}/environments/${profile}-requirements.txt"

echo "环境已准备：${target_path}"
echo "使用方式：${environment_python} experiments/runners/ptq.py list"
