"""张量文件 CLI 的最小端到端测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from ptq_fp4_reference.cli import main


def save_problem(directory: Path) -> tuple[Path, Path]:
    torch.manual_seed(7)
    weight_path = directory / "weight.pt"
    activation_path = directory / "activations.pt"
    torch.save(torch.randn(5, 32), weight_path)
    torch.save(torch.randn(16, 32), activation_path)
    return weight_path, activation_path


def test_mr_gptq_cli_writes_payload(tmp_path: Path) -> None:
    weight_path, activation_path = save_problem(tmp_path)
    output_path = tmp_path / "mr.pt"
    main(
        [
            "mr-gptq",
            "--weight",
            str(weight_path),
            "--activations",
            str(activation_path),
            "--hadamard-group-size",
            "32",
            "--mse-search-steps",
            "3",
            "--output",
            str(output_path),
        ]
    )
    payload = torch.load(output_path, weights_only=True)
    assert payload["method"] == "mr-gptq-reference"
    assert payload["quantized_weight"].shape == (5, 32)


def test_micromix_cli_writes_payload(tmp_path: Path) -> None:
    weight_path, activation_path = save_problem(tmp_path)
    output_path = tmp_path / "micromix.pt"
    main(
        [
            "micromix",
            "--weight",
            str(weight_path),
            "--activations",
            str(activation_path),
            "--output",
            str(output_path),
        ]
    )
    payload = torch.load(output_path, weights_only=True)
    assert payload["method"] == "micromix-reference"
    assert sum(item["end"] - item["start"] for item in payload["partitions"]) == 32


def test_baseline_and_mixfp4_cli_write_payloads(tmp_path: Path) -> None:
    weight_path, activation_path = save_problem(tmp_path)
    baseline_path = tmp_path / "baseline.pt"
    mixfp4_path = tmp_path / "mixfp4.pt"
    common = ["--weight", str(weight_path), "--activations", str(activation_path)]

    main(
        [
            "baseline",
            *common,
            "--baseline",
            "rotation-rtn",
            "--format",
            "nvfp4",
            "--hadamard-group-size",
            "16",
            "--output",
            str(baseline_path),
        ]
    )
    main(["mixfp4", *common, "--output", str(mixfp4_path)])

    baseline = torch.load(baseline_path, weights_only=True)
    mixfp4 = torch.load(mixfp4_path, weights_only=True)
    assert baseline["method"] == "rotation-rtn-nvfp4-baseline"
    assert baseline["quantizer_config"]["scale_strategy"] == "minmax"
    assert mixfp4["method"] == "mixfp4-algorithm-1-reference"
    assert mixfp4["use_e1m2"].shape == (5, 2)
