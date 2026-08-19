from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from _common import model_config, model_path, repository_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 FlexRot-FP4 所需 Hugging Face 模型")
    parser.add_argument("models", nargs="+", help="例如 qwen25_7b qwen3_8b llama31_8b")
    args = parser.parse_args()
    paths = repository_paths()
    for name in args.models:
        config = model_config(name)
        target = model_path(name, paths)
        target.mkdir(parents=True, exist_ok=True)
        print(f"[download] {config['huggingface_id']} -> {target}", flush=True)
        snapshot_download(
            repo_id=config["huggingface_id"],
            local_dir=target,
            token=True if config.get("requires_hf_token") else None,
        )


if __name__ == "__main__":
    main()
