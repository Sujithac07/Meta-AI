"""
Deployment Guard - Self-Healing MLOps Engine
Data drift detection, FastAPI generation, and model versioning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.stats import ks_2samp
import joblib
import json
import hashlib
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DeploymentGuard:
    """
    Self-healing MLOps engine with:
    - Data drift monitoring (KS test)
    - FastAPI wrapper generation
    - Model version registry
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        self.drift_threshold = 0.05  # p-value threshold for drift detection
        self.reference_data = None
        self.reference_fingerprint = None
        
        # Create model directory if needed
        os.makedirs(model_dir, exist_ok=True)
    
    # ==================== DRIFT ENGINE ====================
    
    def set_reference_data(self, df: pd.DataFrame) -> Dict:
        """Set reference data for drift detection."""
        self.reference_data = df.select_dtypes(include=[np.number]).copy()
        self.reference_fingerprint = self._compute_fingerprint(df)
        
        return {
            "status": "success",
            "columns_tracked": list(self.reference_data.columns),
            "samples": len(self.reference_data),
            "fingerprint": self.reference_fingerprint[:16] + "..."
        }
    
    def detect_drift(self, new_data: pd.DataFrame, 
                    significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Detect data drift using Kolmogorov-Smirnov test.
        Compares new data distribution against reference data.
        """
        if self.reference_data is None:
            return {"error": "No reference data set. Call set_reference_data() first."}
        
        new_numeric = new_data.select_dtypes(include=[np.number])
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "samples_checked": len(new_data),
            "significance_level": significance_level,
            "columns_analyzed": [],
            "drifted_columns": [],
            "drift_detected": False,
            "overall_health": "HEALTHY",
            "warnings": []
        }
        
        for col in self.reference_data.columns:
            if col not in new_numeric.columns:
                drift_results["warnings"].append(f"Column '{col}' missing in new data")
                continue
            
            ref_values = self.reference_data[col].dropna().values
            new_values = new_numeric[col].dropna().values
            
            if len(ref_values) < 10 or len(new_values) < 10:
                continue
            
            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(ref_values, new_values)
            
            is_drifted = p_value < significance_level
            
            col_result = {
                "column": col,
                "ks_statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 4),
                "drifted": is_drifted,
                "ref_mean": round(float(np.mean(ref_values)), 4),
                "new_mean": round(float(np.mean(new_values)), 4),
                "mean_shift_pct": round(abs(np.mean(new_values) - np.mean(ref_values)) / (abs(np.mean(ref_values)) + 1e-10) * 100, 2)
            }
            
            drift_results["columns_analyzed"].append(col_result)
            
            if is_drifted:
                drift_results["drifted_columns"].append(col)
                drift_results["drift_detected"] = True
        
        # Determine overall health
        drift_ratio = len(drift_results["drifted_columns"]) / max(len(drift_results["columns_analyzed"]), 1)
        
        if drift_ratio > 0.5:
            drift_results["overall_health"] = "CRITICAL"
            drift_results["recommendation"] = "Retrain model immediately - significant data drift detected"
        elif drift_ratio > 0.2:
            drift_results["overall_health"] = "WARNING"
            drift_results["recommendation"] = "Monitor closely - moderate drift detected in some features"
        elif drift_results["drift_detected"]:
            drift_results["overall_health"] = "CAUTION"
            drift_results["recommendation"] = "Minor drift detected - continue monitoring"
        else:
            drift_results["recommendation"] = "No action needed - data distribution stable"
        
        return drift_results
    
    def _compute_fingerprint(self, df: pd.DataFrame) -> str:
        """Compute data fingerprint for version tracking."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        fingerprint_data = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_stats": {}
        }
        
        for col in numeric_df.columns:
            fingerprint_data["numeric_stats"][col] = {
                "mean": float(numeric_df[col].mean()),
                "std": float(numeric_df[col].std()),
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max())
            }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    # ==================== VERSION REGISTRY ====================
    
    def save_model(self, model: Any, 
                  model_name: str,
                  accuracy: float,
                  training_data: pd.DataFrame,
                  extra_metadata: Dict = None) -> Dict:
        """
        Save model with sidecar metadata file.
        
        Args:
            model: Trained model
            model_name: Name for the model file
            accuracy: Model accuracy/score
            training_data: DataFrame used for training
            extra_metadata: Additional metadata to store
        
        Returns:
            Save result with file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace(" ", "_").lower()
        
        # File paths
        model_file = os.path.join(self.model_dir, f"{safe_name}_{timestamp}.joblib")
        metadata_file = os.path.join(self.model_dir, f"{safe_name}_{timestamp}_metadata.json")
        
        # Compute data fingerprint
        data_fingerprint = self._compute_fingerprint(training_data)
        
        # Build metadata
        metadata = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "model_file": os.path.basename(model_file),
            "training_date": datetime.now().isoformat(),
            "accuracy": round(accuracy, 4),
            "data_fingerprint": data_fingerprint,
            "training_samples": len(training_data),
            "feature_columns": list(training_data.columns),
            "feature_count": len(training_data.columns),
            "version": "1.0.0",
            "framework": "scikit-learn"
        }
        
        # Add extra metadata
        if extra_metadata:
            metadata["extra"] = extra_metadata
        
        # Save model
        joblib.dump(model, model_file)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also set reference data for drift detection
        self.set_reference_data(training_data)
        
        return {
            "status": "success",
            "model_file": model_file,
            "metadata_file": metadata_file,
            "model_size_kb": round(os.path.getsize(model_file) / 1024, 2),
            "fingerprint": data_fingerprint[:16] + "...",
            "accuracy": metadata["accuracy"]
        }
    
    def load_model(self, model_file: str) -> Tuple[Any, Dict]:
        """Load model and its metadata."""
        if not os.path.exists(model_file):
            model_file = os.path.join(self.model_dir, model_file)
        
        if not os.path.exists(model_file):
            return None, {"error": f"Model file not found: {model_file}"}
        
        # Load model
        model = joblib.load(model_file)
        
        # Try to load metadata
        metadata_file = model_file.replace(".joblib", "_metadata.json")
        metadata = {}
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def list_models(self) -> List[Dict]:
        """List all saved models in the registry."""
        models = []
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith("_metadata.json"):
                filepath = os.path.join(self.model_dir, filename)
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                    models.append({
                        "name": metadata.get("model_name"),
                        "type": metadata.get("model_type"),
                        "accuracy": metadata.get("accuracy"),
                        "date": metadata.get("training_date"),
                        "file": metadata.get("model_file")
                    })
        
        return sorted(models, key=lambda x: x.get("date", ""), reverse=True)
    
    # ==================== FASTAPI GENERATOR ====================
    
    def generate_fastapi_app(self, model_file: str, 
                            feature_columns: List[str],
                            output_path: str = None) -> Dict:
        """
        Generate standalone FastAPI app.py with Pydantic validation.
        
        Args:
            model_file: Path to saved model
            feature_columns: List of feature column names
            output_path: Where to save app.py (default: ./api/app.py)
        
        Returns:
            Generation result
        """
        if output_path is None:
            output_path = os.path.join(self.model_dir, "..", "api", "generated_app.py")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate Pydantic model fields
        pydantic_fields = []
        for col in feature_columns:
            clean_name = col.replace(" ", "_").replace("-", "_").lower()
            pydantic_fields.append(f"    {clean_name}: float")
        
        pydantic_model = "\n".join(pydantic_fields)
        
        # Feature mapping for API
        feature_mapping = {col.replace(" ", "_").replace("-", "_").lower(): col for col in feature_columns}
        
        app_code = f'''"""
Auto-Generated FastAPI Model Server
Generated by DeploymentGuard on {datetime.now().isoformat()}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import os

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="Auto-generated prediction API with Pydantic validation",
    version="1.0.0"
)

# Load model
MODEL_PATH = "{model_file}"
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {{MODEL_PATH}}")
    else:
        print(f"WARNING: Model not found at {{MODEL_PATH}}")


# Pydantic schema for request validation
class PredictionRequest(BaseModel):
{pydantic_model}
    
    class Config:
        schema_extra = {{
            "example": {{
                {', '.join([f'"{col.replace(" ", "_").replace("-", "_").lower()}": 0.0' for col in feature_columns[:5]])}
            }}
        }}


class PredictionResponse(BaseModel):
    prediction: float
    probability: float = None
    class_label: str = None
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str = None


# Feature column mapping
FEATURE_MAPPING = {json.dumps(feature_mapping)}
FEATURE_ORDER = {json.dumps(feature_columns)}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_type=type(model).__name__ if model else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction using the trained model.
    All features are validated via Pydantic schema.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        data = request.dict()
        
        # Map API field names to original feature names
        features = {{}}
        for api_name, orig_name in FEATURE_MAPPING.items():
            if api_name in data:
                features[orig_name] = data[api_name]
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features])[FEATURE_ORDER]
        
        # Make prediction
        prediction = float(model.predict(df)[0])
        
        # Get probability if available
        probability = None
        class_label = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            probability = float(max(proba))
            class_label = "Positive" if prediction == 1 else "Negative"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            class_label=class_label,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: list[PredictionRequest]):
    """Batch prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for req in requests:
        try:
            data = req.dict()
            features = {{}}
            for api_name, orig_name in FEATURE_MAPPING.items():
                if api_name in data:
                    features[orig_name] = data[api_name]
            
            df = pd.DataFrame([features])[FEATURE_ORDER]
            prediction = float(model.predict(df)[0])
            
            probability = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)[0]
                probability = float(max(proba))
            
            results.append({{
                "prediction": prediction,
                "probability": probability,
                "status": "success"
            }})
        except Exception as e:
            results.append({{"status": "error", "detail": str(e)}})
    
    return {{"results": results, "total": len(results)}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
'''
        
        # Write the app file
        with open(output_path, 'w') as f:
            f.write(app_code)
        
        return {
            "status": "success",
            "output_path": output_path,
            "endpoints": [
                {"method": "GET", "path": "/health", "description": "Health check"},
                {"method": "POST", "path": "/predict", "description": "Single prediction"},
                {"method": "POST", "path": "/batch_predict", "description": "Batch predictions"}
            ],
            "features_count": len(feature_columns),
            "run_command": f"uvicorn {os.path.basename(output_path).replace('.py', '')}:app --reload"
        }


def format_drift_report(report: Dict[str, Any]) -> str:
    """Format drift report for display."""
    if 'error' in report:
        return f"Error: {report['error']}"
    
    lines = []
    lines.append("=" * 50)
    lines.append("DATA DRIFT ANALYSIS REPORT")
    lines.append("=" * 50)
    
    lines.append(f"\nOverall Health: {report.get('overall_health', 'UNKNOWN')}")
    lines.append(f"Samples Checked: {report.get('samples_checked', 0)}")
    lines.append(f"Drift Detected: {'YES' if report.get('drift_detected') else 'NO'}")
    
    drifted = report.get('drifted_columns', [])
    if drifted:
        lines.append(f"\nDrifted Columns ({len(drifted)}):")
        for col in drifted[:10]:
            lines.append(f"  ! {col}")
    
    lines.append(f"\nRecommendation: {report.get('recommendation', 'N/A')}")
    
    # Show column details
    columns = report.get('columns_analyzed', [])
    if columns:
        lines.append("\nColumn Analysis (top 5 by drift):")
        sorted_cols = sorted(columns, key=lambda x: x.get('p_value', 1))[:5]
        for col in sorted_cols:
            status = "DRIFT" if col.get('drifted') else "OK"
            lines.append(f"  {col['column']}: p={col['p_value']:.4f} [{status}]")
    
    return '\n'.join(lines)
