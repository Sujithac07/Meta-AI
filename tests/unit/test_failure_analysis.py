# ruff: noqa: E402
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.failure_analysis import analyze_failures


def test_analyze_failures_binary_fields_are_present():
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 0]

    out = analyze_failures(y_true, y_pred)

    assert out["total_samples"] == 6
    assert out["total_errors"] == 2
    assert out["false_positives"] == 1
    assert out["false_negatives"] == 1
    assert "recommendations" in out


def test_analyze_failures_multiclass_top_confusions_and_hardest_classes():
    y_true = [0, 1, 2, 0, 1, 2, 2, 1]
    y_pred = [0, 2, 2, 1, 1, 0, 1, 1]

    out = analyze_failures(y_true, y_pred)

    assert out["false_positives"] is None
    assert out["false_negatives"] is None
    assert len(out["top_confusions"]) >= 1
    assert len(out["hardest_classes"]) >= 1
    assert set(out["per_class"].keys()) == {"0", "1", "2"}


def test_analyze_failures_confidence_analysis_for_binary_probability_vector():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    y_proba = [0.1, 0.95, 0.9, 0.1]

    out = analyze_failures(y_true, y_pred, y_proba=y_proba)

    assert out["confidence_analysis"] is not None
    assert out["confidence_analysis"]["overconfident_errors"] >= 1


def test_analyze_failures_handles_length_mismatch():
    out = analyze_failures([1, 0], [1])
    assert "error" in out
    assert out["error_rate"] is None
