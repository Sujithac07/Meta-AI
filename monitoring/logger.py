"""
Centralized logging configuration for MetaAI application.
Handles all logging across training, inference, and pipeline execution.
"""

import logging
import logging.handlers
import os
from utils.config import Config

LOG_DIR = Config.LOGS_DIR
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "app.log")


def setup_logger(name=__name__):
    """
    Configure centralized logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # File handler (rotating, max 10MB per file, keep 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=Config.LOG_MAX_BYTES,
        backupCount=Config.LOG_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Main application logger
app_logger = setup_logger("metaai.app")


def log_training_start(model_name, dataset_shape):
    """Log training start event."""
    app_logger.info(f"Training started: {model_name}, Dataset shape: {dataset_shape}")


def log_training_end(model_name, metrics, execution_time):
    """Log training completion with metrics."""
    f1_score = metrics.get('f1', 0)
    accuracy = metrics.get('accuracy', 0)
    app_logger.info(
        f"Training completed: {model_name}, "
        f"F1={f1_score:.4f}, Accuracy={accuracy:.4f}, "
        f"Time={execution_time:.2f}s"
    )


def log_training_error(model_name, error):
    """Log training error with full traceback."""
    app_logger.error(f"Training failed for {model_name}: {str(error)}", exc_info=True)


def log_pipeline_step(step_name, status, details=""):
    """Log pipeline execution step."""
    msg = f"Pipeline step: {step_name} - {status}"
    if details:
        msg += f" ({details})"
    app_logger.info(msg)


def log_error(function_name, error):
    """Log general error with context."""
    app_logger.error(f"Error in {function_name}: {str(error)}", exc_info=True)


def log_model_performance(model_name, metrics):
    """Log model performance metrics."""
    app_logger.info(
        f"Model Performance - {model_name}: "
        f"Acc={metrics.get('accuracy', 0):.4f}, "
        f"F1={metrics.get('f1', 0):.4f}, "
        f"Precision={metrics.get('precision', 0):.4f}, "
        f"Recall={metrics.get('recall', 0):.4f}"
    )


def log_execution_time(function_name, execution_time):
    """Log function execution time."""
    app_logger.debug(f"{function_name} execution time: {execution_time:.2f}s")


def log_data_loading(num_rows, num_features, filename=""):
    """Log data loading event."""
    msg = f"Data loaded: {num_rows:,} rows, {num_features} features"
    if filename:
        msg += f" from {filename}"
    app_logger.info(msg)


def log_data_error(error, context=""):
    """Log data processing error."""
    msg = f"Data error: {str(error)}"
    if context:
        msg += f" (Context: {context})"
    app_logger.error(msg, exc_info=True)
