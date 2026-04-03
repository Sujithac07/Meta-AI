from core.model_training import train_model
import pandas as pd
from typing import Dict, Any

class ModelAgent:
    """
    Handles model selection logic and training orchestration.
    """
    def __init__(self):
        self.available_models = ["RandomForest", "XGBoost", "LightGBM"]

    def train_candidates(self, df: pd.DataFrame, target: str, suggested_model: str = None) -> Dict[str, Any]:
        """
        Trains the suggested model or defaults to a robust suite.
        """
        results = {}
        model_to_train = suggested_model if suggested_model in self.available_models else "RandomForest"
        
        try:
            model, metrics = train_model(model_to_train, df, target)
            results = {
                "model_name": model_to_train,
                "model_object": model,
                "metrics": metrics
            }
        except Exception as e:
            results = {"error": str(e)}
            
        return results
