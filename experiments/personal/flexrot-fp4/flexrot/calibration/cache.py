from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ActivationCache:
    """逐模块写入 CPU activation cache，并维护可校验 manifest。"""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict] = {}

    def write(self, module_name: str, values: torch.Tensor) -> Path:
        safe_name = module_name.replace(".", "__") + ".pt"
        path = self.root / safe_name
        cpu_values = values.detach().cpu().contiguous()
        torch.save({"module": module_name, "activations": cpu_values}, path)
        self.entries[module_name] = {
            "file": safe_name,
            "rows": int(cpu_values.reshape(-1, cpu_values.shape[-1]).shape[0]),
            "width": int(cpu_values.shape[-1]),
            "dtype": str(cpu_values.dtype),
            "sha256": _sha256(path),
        }
        return path

    def finalize(self, *, metadata: dict) -> Path:
        manifest = {
            "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "storage": "cpu_disk",
            "metadata": metadata,
            "entries": self.entries,
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def load_manifest(path: str | Path, *, verify_files: bool = False) -> dict:
    """读取 calibration manifest；可选逐文件 SHA256 检查断点续跑资产。"""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify_files:
        for entry in payload["entries"].values():
            data_path = manifest_path.parent / entry["file"]
            if _sha256(data_path) != entry["sha256"]:
                raise RuntimeError(f"calibration cache 校验失败：{data_path}")
    return payload
