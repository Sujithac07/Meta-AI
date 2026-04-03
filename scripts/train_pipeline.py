"""
MLOps Training Script with Full Pipeline Integration

This script demonstrates the complete MLOps pipeline:
- Data versioning with DVC
- Experiment tracking with MLflow  
- Model registry
- Automated deployment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from mlops.mlops_pipeline import MLOpsPipeline

def main():
    """Run complete training pipeline"""
    
    print("=" * 70)
    print("  META AI - MLOPS TRAINING PIPELINE")
    print("=" * 70)
    print()
    
    # Initialize pipeline
    pipeline = MLOpsPipeline()
    
    # Load data
    print("[1/4] Loading data...")
    df = pd.read_csv("test_data.csv")
    target_col = df.columns[-1]  # Assume last column is target
    
    print(f"  ✓ Loaded {len(df)} samples with {len(df.columns)} columns")
    print(f"  ✓ Target column: {target_col}")
    print()
    
    # Train multiple models
    models_to_train = [
        ("RandomForest", RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
        ("XGBoost", XGBClassifier, {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss"}),
        ("LightGBM", LGBMClassifier, {"n_estimators": 100, "random_state": 42, "verbose": -1})
    ]
    
    results = []
    
    for i, (name, model_class, params) in enumerate(models_to_train, 1):
        print(f"[2/4] Training model {i}/{len(models_to_train)}: {name}...")
        
        try:
            model, metrics, version = pipeline.run_training_pipeline(
                df=df,
                target_col=target_col,
                model_name=name,
                model_class=model_class,
                params=params
            )
            
            results.append({
                "name": name,
                "version": version,
                "metrics": metrics
            })
            
            print(f"  ✓ {name} trained successfully")
            print()
            
        except Exception as e:
            print(f"  ✗ Error training {name}: {e}")
            print()
    
    # Find best model
    print("[3/4] Selecting best model...")
    best = max(results, key=lambda x: x["metrics"]["accuracy"])
    print(f"  🏆 Best model: {best['name']} v{best['version']}")
    print(f"  📊 Accuracy: {best['metrics']['accuracy']:.4f}")
    print()
    
    # Promote to production
    print("[4/4] Promoting best model to production...")
    pipeline.registry.promote_model(
        model_name=best["name"],
        version=best["version"],
        stage="production"
    )
    print(f"  ✓ {best['name']} v{best['version']} promoted to PRODUCTION")
    print()
    
    print("=" * 70)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    print("  Next steps:")
    print("  • View experiments: mlflow ui")
    print("  • Check registry: cat model_registry/registry.json")
    print("  • Deploy model: Use FastAPI endpoint")
    print()

if __name__ == "__main__":
    main()
