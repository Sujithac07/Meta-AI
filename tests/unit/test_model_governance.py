# ruff: noqa: E402
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_governance import analyze_model_governance


def test_model_governance_with_calibration_signal():
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 0]
    y_proba = [0.05, 0.9, 0.92, 0.15, 0.88, 0.1]

    out = analyze_model_governance(y_true, y_pred, y_proba=y_proba)

    assert out["calibration"] is not None
    assert "expected_calibration_error" in out["calibration"]
    assert "recommendations" in out


def test_model_governance_with_drift_signal():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    reference = [[0.1, 1.0], [0.2, 1.2], [0.15, 0.9], [0.18, 1.1]]
    current = [[0.8, 3.0], [0.75, 2.8], [0.82, 3.2], [0.79, 3.1]]

    out = analyze_model_governance(
        y_true,
        y_pred,
        reference_features=reference,
        current_features=current,
    )

    assert out["drift"] is not None
    assert "feature_psi" in out["drift"]


def test_model_governance_invalid_input():
    out = analyze_model_governance([1, 0], [1])
    assert "error" in out
