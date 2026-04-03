"""Model persistence - save and load trained ML models with joblib."""

import os
import joblib
from typing import Optional, List, Any, Dict, Tuple
from monitoring import app_logger, log_error
from utils.config import Config


def ensure_models_directory():
    """Ensure models directory exists."""
    os.makedirs(Config.MODELS_DIR, exist_ok=True)


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for filename."""
    return model_name.replace(" ", "_").replace("/", "_")


def save_model(model: Any, model_name: str) -> Tuple[bool, str]:
    """Save trained model using joblib."""
    try:
        ensure_models_directory()
        sanitized_name = sanitize_model_name(model_name)
        model_path = os.path.join(Config.MODELS_DIR, f"{sanitized_name}.pkl")
        joblib.dump(model, model_path, compress=Config.MODEL_COMPRESSION)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        app_logger.info(f"Model saved: {model_name} ({file_size_mb:.2f}MB)")
        return True, f"Model '{model_name}' saved successfully"
    except Exception as e:
        msg = f"Failed to save model: {str(e)}"
        log_error(msg)
        return False, msg


def load_model(model_name: str) -> Tuple[Optional[Any], str]:
    """Load saved model using joblib."""
    try:
        sanitized_name = sanitize_model_name(model_name)
        model_path = os.path.join(Config.MODELS_DIR, f"{sanitized_name}.pkl")
        if not os.path.exists(model_path):
            msg = f"Model file not found: {model_name}"
            app_logger.warning(msg)
            return None, msg
        model = joblib.load(model_path)
        app_logger.info(f"Model loaded: {model_name}")
        return model, f"Model '{model_name}' loaded successfully"
    except Exception as e:
        msg = f"Failed to load model: {str(e)}"
        log_error(msg)
        return None, msg


def list_saved_models() -> List[str]:
    """Get list of all saved model names."""
    try:
        ensure_models_directory()
        if not os.path.exists(Config.MODELS_DIR):
            return []
        model_files = [f[:-4] for f in os.listdir(Config.MODELS_DIR) if f.endswith(".pkl")]
        model_names = [f.replace("_", " ") for f in model_files]
        app_logger.debug(f"Found {len(model_names)} saved models")
        return sorted(model_names)
    except Exception as e:
        log_error(f"Failed to list saved models: {str(e)}")
        return []


def delete_model(model_name: str) -> Tuple[bool, str]:
    """Delete a saved model."""
    try:
        sanitized_name = sanitize_model_name(model_name)
        model_path = os.path.join(Config.MODELS_DIR, f"{sanitized_name}.pkl")
        if not os.path.exists(model_path):
            return False, f"Model not found: {model_name}"
        os.remove(model_path)
        app_logger.info(f"Model deleted: {model_name}")
        return True, f"Model '{model_name}' deleted successfully"
    except Exception as e:
        msg = f"Failed to delete model: {str(e)}"
        log_error(msg)
        return False, msg


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a saved model."""
    try:
        sanitized_name = sanitize_model_name(model_name)
        model_path = os.path.join(Config.MODELS_DIR, f"{sanitized_name}.pkl")
        if not os.path.exists(model_path):
            return {}
        stat_info = os.stat(model_path)
        return {
            "name": model_name,
            "path": model_path,
            "size_bytes": stat_info.st_size,
            "size_mb": round(stat_info.st_size / (1024 * 1024), 2),
        }
    except Exception as e:
        log_error(f"Failed to get model info: {str(e)}")
        return {}
