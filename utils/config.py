"""
Centralized configuration for MetaAI application.
All constants and settings managed from this single location.
"""

import os
from typing import List


class Config:
    """
    Centralized configuration for the MetaAI ML platform.
    All constants, paths, and settings defined here.
    """
    
    # ─────────────────────────────────────
    # MODEL CONFIGURATION
    # ─────────────────────────────────────
    DEFAULT_MODELS: List[str] = [
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "LogisticRegression",
        "ExtraTrees",
        "AdaBoost",
        "NaiveBayes",
        "SVC",
        "KNN",
        "HistGradientBoosting",
        "DecisionTree",
    ]
    
    DEFAULT_SELECTED_MODELS: List[str] = [
        "RandomForest",
        "GradientBoosting",
    ]
    
    DEFAULT_TRIALS: int = 10
    MIN_TRIALS: int = 2
    MAX_TRIALS: int = 30
    
    # ─────────────────────────────────────
    # DATA CONFIGURATION
    # ─────────────────────────────────────
    MAX_DATA_ROWS: int = 100000
    MAX_DATA_COLUMNS: int = 500
    MIN_SAMPLES_WARNING: int = 100
    MAX_UPLOAD_SIZE_MB: int = 500
    
    # ─────────────────────────────────────
    # FILE PATHS
    # ─────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    
    # ─────────────────────────────────────
    # LOGGING CONFIGURATION
    # ─────────────────────────────────────
    LOG_LEVEL_FILE: str = "DEBUG"
    LOG_LEVEL_CONSOLE: str = "INFO"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ─────────────────────────────────────
    # MODEL PERSISTENCE
    # ─────────────────────────────────────
    MODEL_SAVE_FORMAT: str = "joblib"
    MODEL_COMPRESSION: int = 3
    
    # ─────────────────────────────────────
    # DATA QUALITY & DRIFT DETECTION
    # ─────────────────────────────────────
    DRIFT_THRESHOLD: float = 0.3
    MISSING_VALUE_THRESHOLD: float = 0.5
    CLASS_IMBALANCE_THRESHOLD: float = 3.0
    DUPLICATE_THRESHOLD: float = 0.01
    
    # ─────────────────────────────────────
    # UI CONFIGURATION
    # ─────────────────────────────────────
    GRADIO_THEME: str = "dark"
    GRADIO_SERVER_NAME: str = "0.0.0.0"  # nosec B104
    GRADIO_SERVER_PORT: int = 7860
    GRADIO_SHARE: bool = False
    GRADIO_DEBUG: bool = False
    
    # ─────────────────────────────────────
    # API KEYS & EXTERNAL SERVICES
    # ─────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # ─────────────────────────────────────
    # PERFORMANCE TUNING
    # ─────────────────────────────────────
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CROSS_VALIDATION_FOLDS: int = 5
    N_JOBS: int = -1
    
    # ─────────────────────────────────────
    # TIMEOUT SETTINGS (seconds)
    # ─────────────────────────────────────
    TRAINING_TIMEOUT: int = 3600
    PREDICTION_TIMEOUT: int = 60
    EDA_TIMEOUT: int = 120
    
    # ─────────────────────────────────────
    # UI & DISPLAY SETTINGS
    # ─────────────────────────────────────
    FEATURE_TOP_N: int = 20
    MISSING_VALUE_TOP_N: int = 15
    LIME_INSTANCE_MAX: int = 500
    REPORT_LINES: int = 30
    REPORT_MAX_LINES: int = 60
    
    # ─────────────────────────────────────
    # PERFORMANCE THRESHOLDS
    # ─────────────────────────────────────
    F1_EXCELLENCE_THRESHOLD: float = 0.8
    F1_GOOD_THRESHOLD: float = 0.7
    F1_DEPLOYMENT_THRESHOLD: float = 0.75
    ACCURACY_DEPLOYMENT_THRESHOLD: float = 0.85
    ACCURACY_CONDITIONAL_THRESHOLD: float = 0.75
    MISSING_PCT_DEPLOYMENT_THRESHOLD: float = 10.0
    CONFIDENCE_THRESHOLD_HIGH: float = 0.8
    CONFIDENCE_THRESHOLD_MEDIUM: float = 0.6
    METRIC_THRESHOLD_EXCELLENT: float = 0.85
    METRIC_THRESHOLD_GOOD: float = 0.70
    
    # ─────────────────────────────────────
    # DATA THRESHOLDS
    # ─────────────────────────────────────
    MODERATE_SAMPLES_THRESHOLD: int = 1000
    MIN_DATA_THRESHOLD: int = 500
    BENCHMARK_MAX_ROWS: int = 5000
    
    # ─────────────────────────────────────
    # QUALITY SCORING PENALTIES
    # ─────────────────────────────────────
    QUALITY_SCORE_MISSING_PENALTY: int = 2
    QUALITY_SCORE_IMBALANCE_PENALTY: int = 15
    QUALITY_SCORE_DUPLICATE_PENALTY: int = 10
    QUALITY_SCORE_ZERO_VARIANCE_PENALTY: int = 3
    QUALITY_SCORE_BALANCE_BONUS: int = 5
    QUALITY_SCORE_COMPLETE_BONUS: int = 5
    QUALITY_SCORE_GOOD: int = 60
    QUALITY_SCORE_EXCELLENT: int = 80
    
    # ─────────────────────────────────────
    # DRIFT DETECTION THRESHOLDS
    # ─────────────────────────────────────
    DRIFT_CRITICAL_THRESHOLD: float = 0.5
    
    # ─────────────────────────────────────
    # SERVER PORTS & ENDPOINTS
    # ─────────────────────────────────────
    MLFLOW_SERVER_PORT: int = 5001
    API_SERVER_PORT: int = 8000
    GRADIO_UI_PORT: int = 7861
    
    # ─────────────────────────────────────
    # API PARAMETERS
    # ─────────────────────────────────────
    OPENAI_MAX_TOKENS: int = 1500
    
    # ─────────────────────────────────────
    # APPLICATION SETTINGS
    # ─────────────────────────────────────
    APP_NAME: str = "MetaAI - Automated Machine Learning Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG_MODE: bool = False
    
    @classmethod
    def get_model_list(cls) -> List[str]:
        """Get the list of available models."""
        return cls.DEFAULT_MODELS
    
    @classmethod
    def get_default_trials(cls, automl: bool = False) -> int:
        """Get default trials for AutoML."""
        return cls.DEFAULT_TRIALS if automl else 1
    
    @classmethod
    def validate_trials(cls, n_trials: int) -> int:
        """Validate and clamp trials value."""
        return max(cls.MIN_TRIALS, min(cls.MAX_TRIALS, n_trials))
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        for directory in [cls.MODELS_DIR, cls.LOGS_DIR, cls.DATA_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)


def load_config_from_env() -> dict:
    """Load config overrides from environment variables."""
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "false").lower() == "true",
        "GRADIO_SHARE": os.getenv("GRADIO_SHARE", "false").lower() == "true",
    }


# Initialize directories on import
Config.ensure_directories()
