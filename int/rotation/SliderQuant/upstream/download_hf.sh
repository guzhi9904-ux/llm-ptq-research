#!/usr/bin/env bash

# Use `huggingface-cli login` first if the model or dataset is gated.
# Replace the model / dataset ids and local directories as needed.

set -euo pipefail

mkdir -p model_zoo datasets_local

# Examples:
# huggingface-cli download --resume-download meta-llama/Llama-2-7b-hf --local-dir model_zoo/Llama-2-7b-hf


