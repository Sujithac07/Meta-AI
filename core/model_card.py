from datetime import datetime

def generate_model_card(
    model_name,
    metrics,
    profile,
    risks,
    stability_results,
    validation_results,
    feature_importance=None
):
    """
    Create a concise model card dict for reporting and governance.
    """
    card = {}
    card["model_name"] = model_name
    card["created_at"] = datetime.utcnow().isoformat() + "Z"
    card["intended_use"] = "Supervised classification for tabular data."
    card["dataset_profile"] = {
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
        "target_distribution": profile.get("target_distribution", {})
    }
    card["validation_checks"] = validation_results.get("checks", {})
    card["validation_warnings"] = validation_results.get("warnings", [])
    card["metrics"] = metrics or {}
    card["stability_summary"] = {
        "mean_f1": stability_results.get(model_name, {}).get("mean"),
        "std_f1": stability_results.get(model_name, {}).get("std"),
        "stability": stability_results.get(model_name, {}).get("stability")
    }
    card["risks"] = {
        "class_imbalance": risks.get("class_imbalance"),
        "imbalance_severity": risks.get("imbalance_severity"),
        "overfitting_risk": risks.get("overfitting_risk"),
        "reasoning": risks.get("reasoning", [])
    }
    card["limitations"] = [
        "Model performance may degrade on out-of-distribution data.",
        "Feature importance may not imply causality."
    ]
    if feature_importance:
        card["top_features"] = list(feature_importance.items())[:10]
    return card
