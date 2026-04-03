import mlflow

def start_experiment(exp_name="Meta-AI-Builder-X"):
    mlflow.set_experiment(exp_name)
    mlflow.start_run()

def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_metrics(metrics: dict):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

def log_decision(decision: str):
    mlflow.log_param("final_decision", decision)

def end_experiment():
    mlflow.end_run()
