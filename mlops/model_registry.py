import mlflow
import mlflow.sklearn

def register_model(model, model_name="MetaAI_Model"):
    # Set tracking URI to local directory to avoid SQLite issues with mlflow-artifacts
    mlflow.set_tracking_uri("./mlruns")
    # Set experiment to ensure it exists
    mlflow.set_experiment("Meta-AI-Models")
    # For local tracking, just log the model without registering
    # This avoids the mlflow-artifacts URI issue with SQLite tracking
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        mlflow.log_param("model_name", model_name)
        print(f"Model logged locally with name: {model_name}")
