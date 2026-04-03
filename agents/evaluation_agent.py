from typing import Dict, Any, List
import pandas as pd

class EvaluationAgent:
    """
    Critically evaluates model performance across multiple dimensions.
    """
    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Interprets metrics and provides a qualitative score.
        """
        acc = metrics.get('accuracy', 0)
        f1 = metrics.get('f1', 0)
        
        status = "POOR"
        if acc > 0.9:
            status = "EXCELLENT"
        elif acc > 0.8:
            status = "GOOD"
        elif acc > 0.7:
            status = "FAIR"
        
        return {
            "status": status,
            "interpretation": f"Model performance is {status} with F1-Score of {f1:.2f}.",
            "recommendation": "Deploy" if acc > 0.8 else "Needs Optimization"
        }

    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Creates a comparison table for the UI."""
        data = []
        for res in results_list:
            row = {"Model": res['model_name']}
            row.update(res['metrics'])
            data.append(row)
        return pd.DataFrame(data)
