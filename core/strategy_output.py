try:
    from core.deep_learning import TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False
import importlib.util

def generate_strategy(risks):

    strategy = {}

    strategy["primary_metric"] = risks["recommended_metric"]

    models_to_try = []
    reasoning = []

    # Include available ML models
    models_to_try.append("LogisticRegression")
    reasoning.append("Logistic Regression as a stable baseline.")

    models_to_try.append("RandomForest")
    reasoning.append("Random Forest to capture non-linear relationships.")

    models_to_try.append("GradientBoosting")
    reasoning.append("Gradient Boosting for ensemble learning.")

    # Restricted to 3 core models as per optimization task
    # try:
    #     import xgboost
    #     models_to_try.append("XGBoost")
    #     reasoning.append("XGBoost for high-performance gradient boosting.")
    # except ImportError:
    #     pass

    # try:
    #     import lightgbm
    #     models_to_try.append("LightGBM")
    #     reasoning.append("LightGBM for efficient gradient boosting.")
    # except ImportError:
    #     pass

    # PyTorch TabNet check
    if TORCH_AVAILABLE:
        try:
            if importlib.util.find_spec("pytorch_tabnet.tab_model") is not None:
                models_to_try.append("TabNet")
                reasoning.append("Deep Learning: TabNet (Attention-based Network) for state-of-the-art tabular learning.")
        except (ImportError, OSError):
            reasoning.append("Note: TabNet dependencies are unavailable in this environment.")
    else:
        reasoning.append("Note: Deep Learning models skipped due to environment initialization issues.")



    strategy["models_to_try"] = models_to_try
    strategy["model_reasoning"] = reasoning

    return strategy
