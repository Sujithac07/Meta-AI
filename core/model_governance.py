from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _to_numpy(values: Any) -> np.ndarray:
    if values is None:
        return np.array([])
    if hasattr(values, "to_numpy"):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Optional[float]:
    if y_proba.size == 0 or y_true.shape[0] != y_proba.shape[0]:
        return None

    if y_proba.ndim == 1:
        confidences = np.where(y_pred == 1, y_proba, 1 - y_proba)
    elif y_proba.ndim == 2:
        confidences = np.max(y_proba, axis=1)
    else:
        return None

    correctness = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        acc = float(np.mean(correctness[in_bin]))
        conf = float(np.mean(confidences[in_bin]))
        weight = float(np.mean(in_bin))
        ece += abs(acc - conf) * weight

    return float(ece)


def _population_stability_index(
    reference_values: np.ndarray,
    current_values: np.ndarray,
    bins: int = 10,
) -> Optional[float]:
    if reference_values.size == 0 or current_values.size == 0:
        return None

    ref = reference_values.astype(float)
    cur = current_values.astype(float)
    if ref.ndim != 1 or cur.ndim != 1:
        return None

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges = np.unique(edges)
    if edges.shape[0] < 2:
        return 0.0

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_hist / max(1, ref_hist.sum())
    cur_pct = cur_hist / max(1, cur_hist.sum())

    eps = 1e-8
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def analyze_model_governance(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    reference_features: Any = None,
    current_features: Any = None,
) -> Dict[str, Any]:
    """
    Evaluate calibration and drift governance signals.

    Returns light-weight, serializable diagnostics for reporting and monitoring.
    """
    y_true_np = _to_numpy(y_true).reshape(-1)
    y_pred_np = _to_numpy(y_pred).reshape(-1)
    y_proba_np = _to_numpy(y_proba)

    if y_true_np.size == 0 or y_pred_np.size == 0 or y_true_np.shape[0] != y_pred_np.shape[0]:
        return {
            "error": "Invalid y_true/y_pred input for governance analysis.",
            "calibration": None,
            "drift": None,
            "recommendations": ["Provide aligned prediction and target vectors."],
        }

    governance: Dict[str, Any] = {
        "calibration": None,
        "drift": None,
        "recommendations": [],
    }

    ece = _expected_calibration_error(y_true_np, y_pred_np, y_proba_np, n_bins=10)
    if ece is not None:
        governance["calibration"] = {
            "expected_calibration_error": round(float(ece), 4),
            "status": "good" if ece < 0.05 else "moderate" if ece < 0.12 else "poor",
        }
        if ece >= 0.12:
            governance["recommendations"].append(
                "Calibration risk is high. Add Platt/Isotonic calibration before deployment."
            )

    if reference_features is not None and current_features is not None:
        ref = _to_numpy(reference_features)
        cur = _to_numpy(current_features)
        if ref.ndim == 2 and cur.ndim == 2 and ref.shape[1] == cur.shape[1]:
            feature_psi = {}
            for idx in range(ref.shape[1]):
                psi = _population_stability_index(ref[:, idx], cur[:, idx], bins=10)
                if psi is not None:
                    feature_psi[str(idx)] = round(float(psi), 4)

            high_drift = {k: v for k, v in feature_psi.items() if v >= 0.25}
            governance["drift"] = {
                "feature_psi": feature_psi,
                "high_drift_features": high_drift,
            }
            if high_drift:
                governance["recommendations"].append(
                    "Data drift detected (PSI >= 0.25). Retrain using recent production slices."
                )

    if not governance["recommendations"]:
        governance["recommendations"].append(
            "Governance checks are stable. Continue scheduled monitoring and periodic recalibration."
        )

    return governance
