from agents.data_agent import DataAgent
from agents.risk_agent import RiskAgent
from agents.strategy_agent import StrategyAgent
from agents.training_agent import TrainingAgent
from agents.orchestrator_agent import OrchestratorAgent
from core.explainability import compute_feature_importance
from mlops.model_registry import register_model
from core.data_validation import validate_dataset
from core.model_card import generate_model_card
from core.report_generator import generate_report
from core.stability_analysis import stability_test
from core.model_training import train_model
from core.failure_analysis import analyze_failures
from core.model_governance import analyze_model_governance
from core.llm_advisor import generate_llm_insights

def run_meta_ai_pipeline(df, target_col):

    # 1. Data profiling
    data_agent = DataAgent()
    data_out = data_agent.run(df, target_col)
    validation_out = validate_dataset(df, target_col)

    # 2. Risk analysis
    risk_agent = RiskAgent()
    risk_out = risk_agent.run(data_out["profile"])

    # 3. Strategy selection
    strategy_agent = StrategyAgent()
    strategy_out = strategy_agent.run(risk_out["risks"])

    # 4. Model training
    training_agent = TrainingAgent()
    training_out = training_agent.run(
        df, target_col, strategy_out["strategy"]
    )

    model = training_out["best_model"]
    metrics = training_out["all_results"]  # Now it's a dict of all results
    best_metrics = training_out["best_metrics"]
    best_model_name = training_out["best_model_name"]

    # 4.1 Stability analysis for selected strategy
    stability_results = {}
    for model_name in strategy_out["strategy"]["models_to_try"]:
        stability_results[model_name] = stability_test(
            train_model, model_name, df, target_col,
            strategy_out["strategy"]["primary_metric"],
            runs=3
        )

    # 5. Explainability (FORCED execution)
    X = df.drop(columns=[target_col])
    feature_importance = compute_feature_importance(model, X)

    # 5.1 Failure analysis
    failure_analysis = {}
    governance_analysis = {}
    try:
        y = df[target_col]
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        failure_analysis = analyze_failures(y, y_pred, y_proba=y_proba)
        governance_analysis = analyze_model_governance(y, y_pred, y_proba=y_proba)
    except Exception:
        failure_analysis = {}
        governance_analysis = {}

    # 5.2 LLM insights (optional)
    llm_insights = generate_llm_insights(
        data_out["profile"],
        risk_out["risks"],
        strategy_out["strategy"],
        metrics,
        validation_out,
        stability_results,
        failure_analysis
    )

    # 6. Orchestration decision
    orchestrator = OrchestratorAgent()
    final_decision = orchestrator.decide(
        data_out, risk_out, strategy_out, validation_out
    )

    # 7. Model versioning
    if final_decision["final_decision"] in ["ACCEPT", "ACCEPT_WITH_WARNINGS"]:
        register_model(model)

    # 8. Model card + report
    model_card = generate_model_card(
        best_model_name,
        best_metrics,
        data_out["profile"],
        risk_out["risks"],
        stability_results,
        validation_out,
        feature_importance
    )

    report_path = generate_report(
        data_out["profile"],
        risk_out["risks"],
        strategy_out,
        metrics,
        stability_results,
        final_decision,
        validation=validation_out,
        model_card=model_card,
        feature_importance=feature_importance,
        failure_analysis=failure_analysis,
        model_governance=governance_analysis,
        llm_insights=llm_insights
    )

    return {
        "data": data_out,
        "risk": risk_out,
        "strategy": strategy_out,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "stability": stability_results,
        "validation": validation_out,
        "model_card": model_card,
        "failure_analysis": failure_analysis,
        "model_governance": governance_analysis,
        "llm_insights": llm_insights,
        "report_path": report_path,
        "final_decision": final_decision
    }
