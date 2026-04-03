from agents.data_agent import DataAgent
from agents.model_agent import ModelAgent
from agents.evaluation_agent import EvaluationAgent
from agents.explanation_agent import ExplanationAgent
from agents.memory_manager import MemoryManager
from core.model_selector_llm import ModelSelectorLLM
from core.business_impact import BusinessImpactModule
import pandas as pd
from typing import Dict, Any


class OrchestratorAgent:
    """
    The High-Level Controller for the Agentic AI system.
    Coordinates all sub-agents and modules.
    """
    def __init__(self):
        self.data_agent = DataAgent()
        self.model_selector = ModelSelectorLLM()
        self.model_agent = ModelAgent()
        self.evaluator = EvaluationAgent()
        self.explainer = ExplanationAgent()
        self.memory = MemoryManager()
        self.business_logic = BusinessImpactModule()

    def run_workflow(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """
        Executes the full Agentic AutoML pipeline.
        """
        cleaned_df, data_summary = self.data_agent.analyze_and_clean(df)

        strategy = self.model_selector.suggest_strategy(data_summary)

        training_result = self.model_agent.train_candidates(
            cleaned_df,
            target,
            suggested_model=strategy["suggested_model"],
        )

        if "error" in training_result:
            return {"error": training_result["error"]}

        evaluation = self.evaluator.evaluate(training_result["metrics"])

        explanation = self.explainer.generate_report(
            training_result["model_object"],
            cleaned_df.drop(columns=[target]).head(10),
        )

        impact = self.business_logic.analyze_impact(
            training_result["metrics"],
            len(cleaned_df),
        )

        self.memory.save_decision(
            dataset_name="Current Upload",
            model_name=training_result["model_name"],
            metrics=training_result["metrics"],
            decision_logic=strategy["reasoning"],
        )

        return {
            "status": evaluation["status"],
            "model_name": training_result["model_name"],
            "metrics": training_result["metrics"],
            "strategy": strategy,
            "evaluation": evaluation,
            "business_impact": impact,
            "explanation_summary": explanation.get("summary", "N/A"),
        }

    def decide(
        self,
        data_out: Dict[str, Any],
        risk_out: Dict[str, Any],
        strategy_out: Dict[str, Any],
        validation_out: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Legacy decision API retained for the existing pipeline/tests."""
        decision = {
            "final_decision": "ACCEPT",
            "reason": "Risks within acceptable limits.",
            "trace": [],
        }

        risks = risk_out.get("risks", {}) if isinstance(risk_out, dict) else {}
        warnings = validation_out.get("warnings", []) if isinstance(validation_out, dict) else []

        if risks.get("overfitting_risk") == "high":
            decision["final_decision"] = "ACCEPT_WITH_WARNINGS"
            decision["reason"] = "High overfitting risk detected."
            decision["trace"].append("Overfitting risk flagged as high.")

        if warnings:
            decision["final_decision"] = "ACCEPT_WITH_WARNINGS"
            decision["reason"] = "Data validation warnings detected."
            decision["trace"].append("Data validation warnings detected.")

        return decision
