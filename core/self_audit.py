def self_audit(risks, eval_results, stability_results, failure_results, validation=None):
    """
    Calculates a Production Readiness Score (PRS) and provides deep-dive audit trails.
    """
    audit = {}
    reasons = []
    warnings = []
    
    score_components = {
        "stability": 0.0,
        "performance": 0.0,
        "fairness": 0.0,
        "data_quality": 0.0
    }

    # 1. Performance Scrutiny
    avg_f1 = sum([m.get('f1', 0) for m in eval_results.values()]) / len(eval_results) if eval_results else 0
    score_components["performance"] = avg_f1 * 100

    # 2. Stability Audit
    stable_count = sum([1 for s in stability_results.values() if s["stability"] == "high"])
    score_components["stability"] = (stable_count / len(stability_results) * 100) if stability_results else 0
    if stable_count < len(stability_results):
        warnings.append(f"Inconsistent performance: Only {stable_count}/{len(stability_results)} models are stable.")

    # 3. Fairness & Bias Audit
    if risks.get("class_imbalance"):
        score_components["fairness"] = 50.0 
        warnings.append("Class imbalance detected: high risk of disparate impact.")
    else:
        score_components["fairness"] = 100.0

    # 4. Data Quality Audit
    if validation and not validation.get("warnings"):
        score_components["data_quality"] = 100.0
    else:
        score_components["data_quality"] = 60.0 
        warnings.append("Data validation warnings: model may be learning noise.")

    # Final PRS Calculation (Weighted Average)
    prs = (score_components["performance"] * 0.4 + 
           score_components["stability"] * 0.3 + 
           score_components["fairness"] * 0.2 + 
           score_components["data_quality"] * 0.1)

    audit["prs_score"] = round(prs, 2)
    
    if prs > 85:
        audit["decision"] = "READY_FOR_PROD"
        reasons.append("Superior stability and metric performance.")
    elif prs > 65:
        audit["decision"] = "ACCEPT_WITH_CAUTION"
        reasons.append("Minor risks in stability or data quality observed.")
    else:
        audit["decision"] = "REJECT_EXPERIMENTAL"
        reasons.append("High risk of failure in production environment.")

    audit["reasons"] = reasons
    audit["warnings"] = warnings
    audit["score_breakdown"] = score_components
    
    return audit
