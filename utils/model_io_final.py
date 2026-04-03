import os
import joblib
from typing import Optional, List, Any
from monitoring import app_logger, log_error

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def ensure_models_directory():
    if not os.path.exists(MODELS_DIR):
        try:
            os.makedirs(MODELS_DIR)
            app_logger.info(f"Created models directory: {MODELS_DIR}")
        except Exception as e:
            app_logger.error(f"Failed to create models directory: {str(e)}")
            raise

def save_model(model: Any, model_name: str) -> tuple[bool, str]:
    try:
        ensure_models_directory()
        if not model:
            return False, "Error: Model is None"
        if not model_name or not isinstance(model_name, str):
            return False, "Error: Invalid model name"
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        joblib.dump(model, model_path, compress=3)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        app_logger.info(f"Model saved: {model_name} ({file_size_mb:.2f}MB) → {model_path}")
        return True, f"Model saved: {model_name}"
    except PermissionError:
        msg = "Error: Permission denied when saving model"
        app_logger.error(msg)
        return False, msg
    except OSError as e:
        msg = f"Error: Disk error when saving model - {str(e)[:100]}"
        app_logger.error(msg)
        return False, msg
    except Exception as e:
        log_error("save_model", e)
        msg = f"Error: Failed to save model - {str(e)[:100]}"
        app_logger.error(msg, exc_info=True)
        return False, msg

def load_model(model_name: str) -> tuple[Optional[Any], str]:
    try:
        if not model_name or not isinstance(model_name, str):
            return None, "Error: Invalid model name"
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        if not os.path.exists(model_path):
            msg = f"Error: Model file not found - {model_name}"
            app_logger.warning(msg)
            return None, msg
        model = joblib.load(model_path)
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        app_logger.info(f"Model loaded: {model_name} ({file_size_mb:.2f}MB) from {model_path}")
        return model, f"Model loaded: {model_name}"
    except FileNotFoundError:
        msg = f"Error: Model file not found - {model_name}"
        app_logger.warning(msg)
        return None, msg
    except EOFError:
        msg = f"Error: Model file corrupted (truncated) - {model_name}"
        app_logger.error(msg)
        return None, msg
    except (joblib.compat.pickle.UnpicklingError, AttributeError):
        msg = f"Error: Model file corrupted (invalid format) - {model_name}"
        app_logger.error(msg)
        return None, msg
    except PermissionError:
        msg = "Error: Permission denied when loading model"
        app_logger.error(msg)
        return None, msg
    except Exception as e:
        log_error("load_model", e)
        msg = f"Error: Failed to load model - {str(e)[:100]}"
        app_logger.error(msg, exc_info=True)
        return None, msg

def list_saved_models() -> List[str]:
    try:
        ensure_models_directory()
        if not os.path.exists(MODELS_DIR):
            return []
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        model_names = [f.replace('.pkl', '').replace('_', ' ') for f in model_files]
        app_logger.debug(f"Found {len(model_names)} saved models: {model_names}")
        return sorted(model_names)
    except PermissionError:
        app_logger.error("Permission denied when listing models")
        return []
    except Exception as e:
        log_error("list_saved_models", e)
        app_logger.error(f"Failed to list saved models: {str(e)}", exc_info=True)
        return []

def delete_model(model_name: str) -> tuple[bool, str]:
    try:
        if not model_name or not isinstance(model_name, str):
            return False, "Error: Invalid model name"
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        if not os.path.exists(model_path):
            return False, f"Error: Model file not found - {model_name}"
        os.remove(model_path)
        app_logger.info(f"Model deleted: {model_name}")
        return True, f"Model deleted: {model_name}"
    except PermissionError:
        msg = "Error: Permission denied when deleting model"
        app_logger.error(msg)
        return False, msg
    except Exception as e:
        log_error("delete_model", e)
        msg = f"Error: Failed to delete model - {str(e)[:100]}"
        app_logger.error(msg, exc_info=True)
        return False, msg

def get_model_info(model_name: str) -> dict:
    try:
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        if not os.path.exists(model_path):
            return {"error": f"Model not found: {model_name}"}
        stat = os.stat(model_path)
        return {
            "name": model_name,
            "path": model_path,
            "size_mb": stat.st_size / (1024 * 1024),
            "size_bytes": stat.st_size,
            "modified_timestamp": stat.st_mtime,
        }
    except Exception as e:
        app_logger.error(f"Failed to get model info: {str(e)}")
        return {"error": str(e)}
