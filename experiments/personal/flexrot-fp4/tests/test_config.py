"""检查 YAML 关键参数完整性和路径模板不会泄漏旧 Windows 绝对路径。"""

from pathlib import Path

from flexrot.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_phase7_config_contains_reproduction_gate() -> None:
    config = load_yaml(ROOT / "configs/experiments/phase7_reproduction.yaml")
    assert config["evaluation"]["expected_predicted_tokens"] == 4088
    assert config["gptq"]["damping"] == 0.01
    assert config["mrgptq"]["static_actorder"] is True
    assert config["expected_results"]["nvfp4_rtn"]["5.625"] == 15.301589


def test_paths_example_is_portable() -> None:
    text = (ROOT / "configs/paths.example.yaml").read_text(encoding="utf-8")
    assert "E:\\graduateStudent" not in text
    assert "/data/models" in text


def test_paper_protocol_separates_baseline_and_flexrot() -> None:
    config = load_yaml(ROOT / "configs/experiments/paper_aligned.yaml")
    assert config["calibration"]["profiles"]["pilot"]["num_sequences"] == 128
    assert config["calibration"]["profiles"]["paper"]["num_sequences"] == 1024
    assert len(config["calibration"]["profiles"]["pilot"]["seeds"]) >= 3
    assert config["baseline"]["angles"] == [0.0, 45.0]
    assert 45.0 not in config["flexrot"]["nvfp4_angles"]
    assert config["quantization"]["mxfp_scale_mode"] == "paper"
