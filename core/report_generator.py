import json

def _pretty(obj):
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)

def generate_report(profile, risks, strategy, eval_results,
                    stability_results, audit, validation=None,
                    model_card=None, feature_importance=None,
                    failure_analysis=None, model_governance=None,
                    llm_insights=None):
    lines = []

    lines.append("META-AI BUILDER++ ENGINEERING REPORT\n")

    lines.append("1. DATASET PROFILE")
    lines.append(_pretty(profile))

    if validation is not None:
        lines.append("\n1.1 DATA VALIDATION")
        lines.append(_pretty(validation))

    lines.append("\n2. IDENTIFIED RISKS")
    lines.append(_pretty(risks))

    lines.append("\n3. MODELING STRATEGY")
    lines.append(_pretty(strategy))

    lines.append("\n4. MODEL EVALUATION RESULTS")
    lines.append(_pretty(eval_results))

    lines.append("\n5. STABILITY & RELIABILITY ANALYSIS")
    lines.append(_pretty(stability_results))

    if feature_importance is not None:
        lines.append("\n5.1 FEATURE IMPORTANCE (TOP FEATURES)")
        try:
            top_feats = list(feature_importance.items())[:10]
        except Exception:
            top_feats = feature_importance
        lines.append(_pretty(top_feats))

    if model_card is not None:
        lines.append("\n5.2 MODEL CARD")
        lines.append(_pretty(model_card))

    if failure_analysis is not None:
        lines.append("\n5.3 FAILURE ANALYSIS")
        lines.append(_pretty(failure_analysis))

    if model_governance is not None:
        lines.append("\n5.4 MODEL GOVERNANCE")
        lines.append(_pretty(model_governance))

    if llm_insights:
        lines.append("\n5.5 LLM INSIGHTS")
        lines.append(str(llm_insights))

    lines.append("\n6. SELF-AUDIT & FINAL DECISION")
    lines.append(_pretty(audit))

    lines.append(
        "\nNOTE: This system prioritizes trust, reliability, and "
        "explainability over raw performance metrics."
    )

    report_text = "\n".join(lines)

    with open("meta_ai_report.txt", "w") as f:
        f.write(report_text)

    return "meta_ai_report.txt"
