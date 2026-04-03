from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def _to_numpy(values: Any) -> np.ndarray:
    """Safely convert array-like inputs to a 1D numpy array."""
    if values is None:
        return np.array([])

    if hasattr(values, "to_numpy"):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)

    if arr.ndim == 0:
        return arr.reshape(1)

    if arr.ndim > 1:
        return arr.reshape(-1)

    return arr


def _build_confidence_scores(y_pred: np.ndarray, y_proba: np.ndarray) -> Optional[np.ndarray]:
    """
    Build per-sample confidence for predicted labels.

    Supported shapes:
    - (n_samples,) -> interpreted as positive-class probability for binary prediction.
    - (n_samples, n_classes) -> max probability for predicted class.
    """
    if y_proba.size == 0:
        return None

    if y_proba.ndim == 1:
        confidence = np.where(y_pred == 1, y_proba, 1 - y_proba)
        return np.clip(confidence, 0.0, 1.0)

    if y_proba.ndim == 2:
        if y_proba.shape[0] != y_pred.shape[0]:
            return None
        max_proba = np.max(y_proba, axis=1)
        return np.clip(max_proba, 0.0, 1.0)

    return None


def _normalize_confusion_matrix(cm: np.ndarray) -> List[List[float]]:
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(cm, row_sums, where=row_sums != 0)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized.round(4).tolist()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def analyze_failures(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
    y_proba: Optional[Sequence[Any]] = None,
    top_k_confusions: int = 3,
) -> Dict[str, Any]:
    """
    Compute rich failure diagnostics for classification outputs.

    Backward-compatible with previous interface while adding:
    - class-level error topology,
    - dominant confusion pairs,
    - confidence-risk diagnostics,
    - actionable recommendations.
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    if y_true_np.size == 0 or y_pred_np.size == 0:
        return {
            "error": "Empty y_true or y_pred.",
            "total_samples": 0,
            "error_rate": None,
            "accuracy": None,
            "recommendations": [
                "Provide non-empty classification targets and predictions."
            ],
        }

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        return {
            "error": "y_true and y_pred lengths do not match.",
            "total_samples": int(y_true_np.shape[0]),
            "error_rate": None,
            "accuracy": None,
            "recommendations": [
                "Align prediction output size with target vector before analysis."
            ],
        }

    label_space = list(labels) if labels is not None else sorted(
        set(y_true_np.tolist()) | set(y_pred_np.tolist())
    )

    cm = confusion_matrix(y_true_np, y_pred_np, labels=label_space)
    report = classification_report(
        y_true_np,
        y_pred_np,
        labels=label_space,
        output_dict=True,
        zero_division=0,
    )

    total_samples = int(y_true_np.shape[0])
    total_errors = int(np.sum(y_true_np != y_pred_np))
    accuracy = _safe_float(np.mean(y_true_np == y_pred_np))
    error_rate = _safe_float(np.mean(y_true_np != y_pred_np))

    failures: Dict[str, Any] = {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "labels": [str(x) for x in label_space],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": _normalize_confusion_matrix(cm),
        "classification_report": report,
    }

    if cm.shape == (2, 2):
        failures["true_negatives"] = int(cm[0][0])
        failures["false_positives"] = int(cm[0][1])
        failures["false_negatives"] = int(cm[1][0])
        failures["true_positives"] = int(cm[1][1])
    else:
        failures["true_negatives"] = None
        failures["false_positives"] = None
        failures["false_negatives"] = None
        failures["true_positives"] = None

    per_class: Dict[str, Dict[str, Any]] = {}
    for i, label in enumerate(label_space):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - cm[i, i])
        fn = int(cm[i, :].sum() - cm[i, i])
        support = int(cm[i, :].sum())

        precision = report.get(str(label), {}).get("precision", 0.0)
        recall = report.get(str(label), {}).get("recall", 0.0)
        f1 = report.get(str(label), {}).get("f1-score", 0.0)

        miss_rate = (fn / support) if support else 0.0
        false_discovery_rate = (fp / (tp + fp)) if (tp + fp) else 0.0

        per_class[str(label)] = {
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "miss_rate": round(float(miss_rate), 4),
            "false_discovery_rate": round(float(false_discovery_rate), 4),
        }

    failures["per_class"] = per_class

    confusion_pairs = []
    for i, actual_label in enumerate(label_space):
        for j, predicted_label in enumerate(label_space):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                confusion_pairs.append(
                    {
                        "actual": str(actual_label),
                        "predicted": str(predicted_label),
                        "count": count,
                    }
                )

    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    failures["top_confusions"] = confusion_pairs[: max(1, int(top_k_confusions))]

    hardest_classes = sorted(
        (
            {
                "class": label,
                "error_burden": stats["false_positives"] + stats["false_negatives"],
                "miss_rate": stats["miss_rate"],
                "precision": stats["precision"],
                "recall": stats["recall"],
            }
            for label, stats in per_class.items()
        ),
        key=lambda x: (x["error_burden"], x["miss_rate"]),
        reverse=True,
    )
    failures["hardest_classes"] = hardest_classes[:3]

    y_proba_np = _to_numpy(y_proba)
    confidence_scores = _build_confidence_scores(y_pred_np, y_proba_np)
    confidence_analysis: Optional[Dict[str, Any]] = None

    if confidence_scores is not None and confidence_scores.shape[0] == total_samples:
        wrong_mask = y_true_np != y_pred_np
        overconfident_wrong = int(np.sum((confidence_scores >= 0.8) & wrong_mask))
        uncertain_wrong = int(np.sum((confidence_scores <= 0.6) & wrong_mask))
        mean_conf = _safe_float(np.mean(confidence_scores))

        confidence_analysis = {
            "mean_prediction_confidence": round(mean_conf, 4) if mean_conf is not None else None,
            "overconfident_errors": overconfident_wrong,
            "uncertain_errors": uncertain_wrong,
            "overconfident_error_rate": round(
                (overconfident_wrong / total_errors), 4
            ) if total_errors else 0.0,
        }

    failures["confidence_analysis"] = confidence_analysis

    recommendations: List[str] = []
    if error_rate is not None and error_rate >= 0.2:
        recommendations.append(
            "High global error rate detected. Revisit feature engineering and class balance strategy."
        )

    if hardest_classes:
        worst_class = hardest_classes[0]
        if worst_class["error_burden"] > 0:
            recommendations.append(
                "Prioritize class '{}' for targeted remediation (thresholding, class weights, or new features).".format(
                    worst_class["class"]
                )
            )

    if failures["top_confusions"]:
        top_pair = failures["top_confusions"][0]
        recommendations.append(
            "Largest confusion path is '{}' -> '{}'. Audit feature separability for these classes.".format(
                top_pair["actual"], top_pair["predicted"]
            )
        )

    if confidence_analysis and confidence_analysis["overconfident_error_rate"] >= 0.5:
        recommendations.append(
            "Model is frequently wrong with high confidence. Add calibration (Platt/Isotonic/temperature scaling)."
        )

    if not recommendations:
        recommendations.append(
            "Failure profile is controlled. Continue monitoring drift and retrain on fresh data slices."
        )

    failures["recommendations"] = recommendations
    return failures
