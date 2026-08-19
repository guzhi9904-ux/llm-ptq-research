import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_paper_results.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("summarize_paper_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_numeric_metrics_collects_ppl_and_openllm_scores() -> None:
    metrics = MODULE._numeric_metrics(
        {
            "ppl": 10.5,
            "openllm_results": {
                "hellaswag": {"acc_norm,none": 0.7, "acc_norm_stderr": 0.01}
            },
        }
    )
    assert metrics["wikitext2.ppl"] == 10.5
    assert metrics["hellaswag.acc_norm,none"] == 0.7
