from __future__ import annotations

from datasets import load_dataset

from _common import dataset_path, repository_paths


def main() -> None:
    paths = repository_paths()
    target = dataset_path(paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    dataset.save_to_disk(str(target))
    print(f"WikiText-2 已保存到 {target}")


if __name__ == "__main__":
    main()
