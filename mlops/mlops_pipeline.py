# ================================================================================
# META AI BUILDER++ - COMPLETE MLOPS PIPELINE
# ================================================================================
# Based on industry best practices:
# - MLflow for experiment tracking
# - DVC for data versioning
# - GitHub Actions for CI/CD
# - FastAPI for model serving
# - Automated training & deployment
# ================================================================================

"""
End-to-End MLOps Pipeline

Why it matters: In the industry, "the model" is only 5% of the code.
The rest is plumbing. This project teaches you how to automate training,
track experiments, and deploy ML models in production.

Key Concepts:
- MLflow (experiment tracking)
- DVC (data versioning)
- GitHub Actions (CI/CD)
- FastAPI (model serving)
- Automated workflows

This stops you from losing track of which version of your code produced which result.
"""

# ================================================================================
# 1. EXPERIMENT TRACKING - MLFLOW
# ================================================================================

import os
import json
import hashlib
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd

class ExperimentTracker:
    """
    MLflow-based experiment tracking for all ML experiments
    
    Tracks:
    - Model parameters
    - Training metrics
    - Dataset version
    - Code version
    - Artifacts (models, plots, etc.)
    """
    
    def __init__(self, experiment_name="meta_ai_experiments"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.run = None
        
    def start_run(self, run_name=None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name)
        return self.run
    
    def log_params(self, params: dict):
        """Log model parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict, step=None):
        """Log training/validation metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_dataset_info(self, df, target_col):
        """Log dataset information for reproducibility"""
        dataset_info = {
            "n_samples": len(df),
            "n_features": len(df.columns) - 1,
            "target_column": target_col,
            "feature_names": ",".join([c for c in df.columns if c != target_col]),
            "dataset_hash": self._hash_dataset(df)
        }
        mlflow.log_params(dataset_info)
    
    def log_model(self, model, model_name):
        """Log trained model"""
        mlflow.sklearn.log_model(model, model_name)
    
    def log_artifact(self, file_path, artifact_path=None):
        """Log any artifact (plots, reports, etc.)"""
        mlflow.log_artifact(file_path, artifact_path)
    
    def end_run(self):
        """End current MLflow run"""
        if self.run:
            mlflow.end_run()
    
    @staticmethod
    def _hash_dataset(df):
        """Create hash of dataset for versioning"""
        raw = pd.util.hash_pandas_object(df).values
        return hashlib.md5(raw.tobytes(), usedforsecurity=False).hexdigest()


# ================================================================================
# 2. DATA VERSIONING - DVC
# ================================================================================

class DataVersionController:
    """
    DVC-like data versioning system
    
    Tracks:
    - Data files
    - Model files
    - Preprocessing pipelines
    - Dataset transformations
    """
    
    def __init__(self, dvc_dir=".dvc"):
        self.dvc_dir = dvc_dir
        os.makedirs(dvc_dir, exist_ok=True)
        
    def add(self, file_path, metadata=None):
        """Add file to version control"""
        file_hash = self._hash_file(file_path)
        
        # Create .dvc metadata file
        dvc_info = {
            "file": file_path,
            "hash": file_hash,
            "size": os.path.getsize(file_path),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        dvc_file = file_path + ".dvc"
        with open(dvc_file, 'w') as f:
            json.dump(dvc_info, f, indent=2)
        
        return file_hash
    
    def pull(self, dvc_file):
        """Retrieve specific version of data"""
        with open(dvc_file, 'r') as f:
            dvc_info = json.load(f)
        return dvc_info
    
    @staticmethod
    def _hash_file(file_path):
        """Create hash of file for versioning"""
        hash_md5 = hashlib.md5(usedforsecurity=False)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


# ================================================================================
# 3. MODEL REGISTRY
# ================================================================================

class ModelRegistry:
    """
    Centralized model registry for production deployment
    
    Manages:
    - Model versions
    - Model staging (dev/staging/production)
    - Model metadata
    - Performance metrics
    """
    
    def __init__(self, registry_dir="model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.registry_file = os.path.join(registry_dir, "registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load existing registry"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": {}}
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name, model_path, metrics, metadata=None):
        """Register a new model version"""
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {"versions": []}
        
        version = len(self.registry["models"][model_name]["versions"]) + 1
        
        model_info = {
            "version": version,
            "path": model_path,
            "metrics": metrics,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "stage": "dev"  # dev/staging/production
        }
        
        self.registry["models"][model_name]["versions"].append(model_info)
        self._save_registry()
        
        return version
    
    def promote_model(self, model_name, version, stage):
        """Promote model to different stage"""
        if model_name in self.registry["models"]:
            for v in self.registry["models"][model_name]["versions"]:
                if v["version"] == version:
                    v["stage"] = stage
                    self._save_registry()
                    return True
        return False
    
    def get_production_model(self, model_name):
        """Get current production model"""
        if model_name in self.registry["models"]:
            for v in self.registry["models"][model_name]["versions"]:
                if v["stage"] == "production":
                    return v
        return None


# ================================================================================
# 4. MLOPS PIPELINE ORCHESTRATOR
# ================================================================================

class MLOpsPipeline:
    """
    Complete MLOps pipeline orchestrator
    
    Orchestrates:
    - Data ingestion & versioning
    - Model training
    - Experiment tracking
    - Model registration
    - Model deployment
    """
    
    def __init__(self):
        self.tracker = ExperimentTracker()
        self.dvc = DataVersionController()
        self.registry = ModelRegistry()
    
    def run_training_pipeline(self, df, target_col, model_name, model_class, params=None):
        """
        Complete training pipeline with full MLOps tracking
        
        Steps:
        1. Version the data
        2. Start experiment tracking
        3. Train model
        4. Log everything
        5. Register model
        6. Return results
        """
        
        # Step 1: Version data
        data_file = "temp_data.csv"
        df.to_csv(data_file, index=False)
        data_hash = self.dvc.add(data_file, {"target": target_col})
        
        # Step 2: Start experiment tracking
        run = self.tracker.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        try:
            # Step 3: Log dataset info
            self.tracker.log_dataset_info(df, target_col)
            
            # Step 4: Train model
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize model
            if params:
                model = model_class(**params)
            else:
                model = model_class()
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Step 5: Log metrics and model
            self.tracker.log_metrics(metrics)
            if params:
                self.tracker.log_params(params)
            self.tracker.log_model(model, model_name)
            
            # Step 6: Register model
            model_path = f"models/{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.makedirs("models", exist_ok=True)
            
            import joblib
            joblib.dump(model, model_path)
            
            version = self.registry.register_model(
                model_name=model_name,
                model_path=model_path,
                metrics=metrics,
                metadata={
                    "data_hash": data_hash,
                    "mlflow_run_id": run.info.run_id
                }
            )
            
            print(f"✅ Model registered: {model_name} v{version}")
            print(f"📊 Metrics: {metrics}")
            print(f"📦 Saved to: {model_path}")
            
            return model, metrics, version
            
        finally:
            # Step 7: End tracking
            self.tracker.end_run()
            
            # Cleanup
            if os.path.exists(data_file):
                os.remove(data_file)


# ================================================================================
# 5. CI/CD AUTOMATION
# ================================================================================

def setup_github_actions():
    """
    Create GitHub Actions workflow for CI/CD
    
    This automates:
    - Testing
    - Training
    - Deployment
    """
    
    workflow = """
name: ML Pipeline CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Run tests
        run: pytest tests/
  
  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train models
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python scripts/train_pipeline.py
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: models
          path: models/
  
  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Add your deployment commands here
"""
    
    os.makedirs(".github/workflows", exist_ok=True)
    with open(".github/workflows/ml_pipeline.yml", 'w') as f:
        f.write(workflow)
    
    print("✅ GitHub Actions workflow created at .github/workflows/ml_pipeline.yml")


# ================================================================================
# 6. PRODUCTION MONITORING
# ================================================================================

class ModelMonitor:
    """
    Monitor models in production
    
    Tracks:
    - Prediction latency
    - Data drift
    - Model performance degradation
    - Alerts
    """
    
    def __init__(self):
        self.metrics = []
    
    def log_prediction(self, input_data, prediction, latency):
        """Log production prediction"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_hash": hashlib.md5(
                str(input_data).encode(),
                usedforsecurity=False,
            ).hexdigest(),
            "prediction": prediction,
            "latency_ms": latency * 1000
        }
        self.metrics.append(log_entry)
    
    def check_drift(self, reference_data, current_data):
        """Check for data drift"""
        # Simple drift detection using statistical tests
        from scipy.stats import ks_2samp
        
        drift_detected = False
        drift_features = []
        
        for col in reference_data.columns:
            if reference_data[col].dtype in ['int64', 'float64']:
                statistic, pvalue = ks_2samp(reference_data[col], current_data[col])
                if pvalue < 0.05:  # Significant drift
                    drift_detected = True
                    drift_features.append(col)
        
        return drift_detected, drift_features


# ================================================================================
# USAGE EXAMPLE
# ================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  META AI BUILDER++ - MLOPS PIPELINE")
    print("=" * 70)
    print()
    print("  This module provides:")
    print("  ✅ MLflow experiment tracking")
    print("  ✅ DVC data versioning")
    print("  ✅ Model registry")
    print("  ✅ MLOps pipeline orchestration")
    print("  ✅ CI/CD automation")
    print("  ✅ Production monitoring")
    print()
    print("  Import and use in your training scripts!")
    print("=" * 70)
