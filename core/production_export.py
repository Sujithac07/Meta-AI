"""
Production Export - One-Click Deployment Package
Bundles model, preprocessors, API, and requirements for deployment
"""

import os
import shutil
import zipfile
import json
import re
import joblib
import importlib.metadata
from typing import Dict, Any, List, Optional
from datetime import datetime


class ProductionExporter:
    """
    Creates deployment-ready packages with:
    - Trained model
    - Preprocessing scalars/encoders
    - FastAPI wrapper
    - requirements.txt
    - Dockerfile
    - README
    """
    
    # Base requirements for any model (defaults; we try to pin exact versions at export time)
    BASE_REQUIREMENTS = [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
    ]
    
    # Model-specific requirements
    MODEL_REQUIREMENTS = {
        "XGBClassifier": ["xgboost>=2.0.0"],
        "XGBRegressor": ["xgboost>=2.0.0"],
        "LGBMClassifier": ["lightgbm>=4.0.0"],
        "LGBMRegressor": ["lightgbm>=4.0.0"],
        "CatBoostClassifier": ["catboost>=1.2"],
        "CatBoostRegressor": ["catboost>=1.2"],
        "StackingClassifier": ["xgboost>=2.0.0", "lightgbm>=4.0.0"],
        "StackingRegressor": ["xgboost>=2.0.0", "lightgbm>=4.0.0"],
    }
    
    def __init__(self, export_dir: str = "./exports"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_production_package(self,
                                  model: Any,
                                  feature_columns: List[str],
                                  target_column: str,
                                  task_type: str = "classification",
                                  model_name: str = "metaai_model",
                                  accuracy: float = 0.0,
                                  preprocessors: Dict = None,
                                  pydantic_schema_code: Optional[str] = None,
                                  extra_files: Dict[str, str] = None) -> str:
        """
        Create complete production deployment package.
        
        Args:
            model: Trained model object
            feature_columns: List of feature column names
            target_column: Target column name
            task_type: 'classification' or 'regression'
            model_name: Name for the exported model
            accuracy: Model accuracy/score
            preprocessors: Dict of preprocessing objects (scalers, encoders)
            extra_files: Dict of {filename: content} for additional files
        
        Returns:
            Path to the generated zip file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name)).strip("._") or "metaai_model"
        package_name = f"{safe_model_name}_{timestamp}"
        package_dir = os.path.join(self.export_dir, package_name)
        
        # Create package directory
        os.makedirs(package_dir, exist_ok=True)
        os.makedirs(os.path.join(package_dir, "model"), exist_ok=True)
        # Keep everything runnable from the package root (avoid `api.py` vs `api/` import collisions).
        
        # 1. Save model
        model_path = os.path.join(package_dir, "model", "model.joblib")
        joblib.dump(model, model_path)

        # Also export at package root for easier consumption by ops teams.
        # (FastAPI code still loads from model/model.joblib.)
        joblib.dump(model, os.path.join(package_dir, "model.joblib"))

        # 2. Save preprocessors
        if preprocessors:
            for name, preprocessor in preprocessors.items():
                prep_path = os.path.join(package_dir, "model", f"{name}.joblib")
                joblib.dump(preprocessor, prep_path)
        
        # 3. Save model metadata
        metadata = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "task_type": task_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "accuracy": accuracy,
            "created_at": datetime.now().isoformat(),
            "preprocessors": list(preprocessors.keys()) if preprocessors else [],
            "version": "1.0.0"
        }
        
        with open(
            os.path.join(package_dir, "model", "metadata.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2)
        
        # 4. Generate requirements.txt
        requirements = self._generate_requirements(model)
        with open(os.path.join(package_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(requirements))

        # 4.5 Write schema.py (Pydantic BaseModel)
        schema_code = pydantic_schema_code
        if not schema_code:
            # Fallback schema: allow float inputs for all feature columns.
            field_lines = []
            for col in feature_columns:
                clean_name = col.replace(" ", "_").replace("-", "_").lower()
                field_lines.append(f"    {clean_name}: float")
            schema_code = (
                "from pydantic import BaseModel, Field, validator\n"
                "from typing import Optional\n\n"
                "class DataRowSchema(BaseModel):\n"
                '    """Auto-generated Pydantic schema for data validation."""\n\n'
                + "\n".join(field_lines)
                + "\n\n    class Config:\n        extra = 'forbid'\n"
            )
        with open(os.path.join(package_dir, "schema.py"), "w", encoding="utf-8") as f:
            f.write(schema_code)

        # 4.6 Write main.py wrapper (uvicorn target)
        main_py = (
            "from api import app\n\n"
            "if __name__ == '__main__':\n"
            "    import uvicorn\n"
            "    uvicorn.run(app, host='0.0.0.0', port=8000)\n"
        )
        with open(os.path.join(package_dir, "main.py"), "w", encoding="utf-8") as f:
            f.write(main_py)
        
        # 5. Generate FastAPI app (root `api.py`)
        api_code = self._generate_fastapi_app(feature_columns, task_type, preprocessors)
        with open(os.path.join(package_dir, "api.py"), "w", encoding="utf-8") as f:
            f.write(api_code)
        
        # 6. Generate Dockerfile
        dockerfile = self._generate_dockerfile()
        with open(os.path.join(package_dir, "Dockerfile"), "w", encoding="utf-8") as f:
            f.write(dockerfile)
        
        # 7. Generate docker-compose.yml
        compose = self._generate_docker_compose(model_name)
        with open(os.path.join(package_dir, "docker-compose.yml"), "w", encoding="utf-8") as f:
            f.write(compose)
        
        # 8. Generate README
        readme = self._generate_readme(model_name, metadata, requirements)
        with open(os.path.join(package_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme)
        
        # 9. Generate run scripts
        self._generate_run_scripts(package_dir)
        
        # 10. Add extra files if provided
        if extra_files:
            for filename, content in extra_files.items():
                filepath = os.path.join(package_dir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
        
        # 11. Create zip file
        zip_path = f"{package_dir}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, package_dir)
                    zipf.write(file_path, arcname)
        
        # Clean up directory (keep only zip)
        shutil.rmtree(package_dir)
        
        return os.path.abspath(zip_path)
    
    def _generate_requirements(self, model: Any) -> List[str]:
        """Generate requirements.txt based on model type."""
        # Prefer exact pins for core libs to avoid sklearn/joblib incompatibilities
        # between training and serving environments.
        def _pin(pkg: str, fallback: str) -> str:
            try:
                v = importlib.metadata.version(pkg)
                return f"{pkg}=={v}"
            except Exception:
                return fallback

        base = {
            "fastapi": _pin("fastapi", "fastapi>=0.104.0"),
            "uvicorn": _pin("uvicorn", "uvicorn>=0.24.0"),
            "pydantic": _pin("pydantic", "pydantic>=2.0.0"),
            "pandas": _pin("pandas", "pandas>=2.0.0"),
            "numpy": _pin("numpy", "numpy>=1.24.0"),
            "scikit-learn": _pin("scikit-learn", "scikit-learn>=1.3.0"),
            "joblib": _pin("joblib", "joblib>=1.3.0"),
        }
        requirements = list(base.values())
        
        model_type = type(model).__name__
        
        # Check nested estimators (supports both tuples and raw estimator objects)
        if hasattr(model, 'estimators_'):
            for est_entry in model.estimators_:
                estimator = est_entry[1] if isinstance(est_entry, tuple) and len(est_entry) >= 2 else est_entry
                est_type = type(estimator).__name__
                if est_type in self.MODEL_REQUIREMENTS:
                    for req in self.MODEL_REQUIREMENTS[est_type]:
                        if req not in requirements:
                            requirements.append(req)
        
        # Check model type
        if model_type in self.MODEL_REQUIREMENTS:
            for req in self.MODEL_REQUIREMENTS[model_type]:
                if req not in requirements:
                    requirements.append(req)
        
        return sorted(set(requirements))
    
    def _generate_fastapi_app(self, 
                             feature_columns: List[str],
                             task_type: str,
                             preprocessors: Dict = None) -> str:
        """Generate FastAPI application code."""
        # Pydantic request model is imported from `schema.py` (written during export).
        
        # Feature mapping
        feature_mapping = {
            col.replace(" ", "_").replace("-", "_").lower(): col 
            for col in feature_columns
        }
        
        # Preprocessor loading code
        preprocessor_code = ""
        if preprocessors:
            preprocessor_code = """
# Load preprocessors
for prep_file in Path("model").glob("*.joblib"):
    if prep_file.name != "model.joblib":
        name = prep_file.stem
        preprocessors[name] = joblib.load(prep_file)
"""
        
        app_code = f'''"""
MetaAI Pro - Production API Server
Auto-generated deployment endpoint
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
from schema import DataRowSchema

# Initialize FastAPI
app = FastAPI(
    title="MetaAI Pro - Model API",
    description="Production ML prediction endpoint",
    version="1.0.0"
)

# Load model and metadata
MODEL_PATH = Path("model/model.joblib")
METADATA_PATH = Path("model/metadata.json")

model = None
metadata = {{}}
preprocessors = {{}}
{preprocessor_code}

@app.on_event("startup")
async def load_model():
    global model, metadata
    
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded: {{type(model).__name__}}")
    
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        print(f"Metadata loaded: {{metadata.get('model_name')}}")


# Request schema
class PredictionRequest(DataRowSchema):
    pass


class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[float] = None
    label: Optional[str] = None
    confidence: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    instances: List[PredictionRequest]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    accuracy: Optional[float] = None


# Feature configuration
FEATURE_MAPPING = {json.dumps(feature_mapping, indent=4)}
FEATURE_ORDER = {json.dumps(feature_columns)}


def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    """Convert request to DataFrame with correct feature order."""
    data = request.dict()
    features = {{}}
    
    for api_name, orig_name in FEATURE_MAPPING.items():
        if api_name in data:
            features[orig_name] = data[api_name]
    
    return pd.DataFrame([features])[FEATURE_ORDER]


@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint."""
    return {{
        "name": "MetaAI Pro Model API",
        "version": "1.0.0",
        "status": "running"
    }}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_type=type(model).__name__ if model else None,
        accuracy=metadata.get("accuracy")
    )


@app.get("/metadata")
async def get_metadata():
    """Get model metadata."""
    return metadata


def _maybe_transform(df: pd.DataFrame):
    \"\"\"Apply common preprocessors if present (e.g., scaler).\"\"\"
    if not preprocessors:
        return df
    scaler = preprocessors.get("scaler") or preprocessors.get("standardscaler") or preprocessors.get("standard_scaler")
    if scaler is not None:
        return scaler.transform(df)
    return df


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = prepare_features(request)
        X = _maybe_transform(df)
        prediction = float(model.predict(X)[0])
        
        probability = None
        confidence = None
        label = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            confidence = float(max(proba))
            label = "Positive" if prediction == 1 else "Negative"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            label=label,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for instance in request.instances:
        try:
            df = prepare_features(instance)
            X = _maybe_transform(df)
            pred = float(model.predict(X)[0])
            
            prob = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            
            results.append({{"prediction": pred, "probability": prob, "status": "success"}})
        except Exception as e:
            results.append({{"status": "error", "detail": str(e)}})
    
    return {{"results": results, "total": len(results)}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        return app_code
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for deployment."""
        return '''# MetaAI Pro - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    def _generate_docker_compose(self, model_name: str) -> str:
        """Generate docker-compose.yml."""
        return f'''# MetaAI Pro - Docker Compose
version: '3.8'

services:
  api:
    build: .
    container_name: {model_name}-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
'''
    
    def _generate_readme(self, model_name: str, metadata: Dict, requirements: List[str]) -> str:
        """Generate README.md with deployment instructions."""
        return f'''# {model_name} - Production Deployment Package

## Model Information
- **Type**: {metadata.get('model_type', 'Unknown')}
- **Task**: {metadata.get('task_type', 'classification')}
- **Accuracy**: {metadata.get('accuracy', 0):.4f}
- **Features**: {metadata.get('feature_count', 0)}
- **Created**: {metadata.get('created_at', 'Unknown')}

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Build and run
docker-compose up --build

# API available at http://localhost:8000
```

### Option 2: Local Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python main.py

# Or with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Option 3: Windows
```bash
# Double-click run.bat
# Or run in terminal:
run.bat
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/metadata` | GET | Model metadata |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

## Example Request

```bash
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"feature1": 0.5, "feature2": 1.2}}'
```

## Files Included

```
{model_name}/
├── model/
│   ├── model.joblib      # Trained model
│   └── metadata.json     # Model metadata
├── schema.py             # Pydantic request schema
├── main.py               # FastAPI uvicorn entrypoint
├── api/
│   └── app.py           # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker build file
├── docker-compose.yml  # Docker compose config
├── run.bat            # Windows run script
├── run.sh             # Linux/Mac run script
└── README.md          # This file
```

## Requirements

```
{chr(10).join(requirements)}
```

---
Generated by MetaAI Pro - Enterprise AutoML Platform
'''
    
    def _generate_run_scripts(self, package_dir: str):
        """Generate platform-specific run scripts."""
        
        # Windows batch file
        bat_content = '''@echo off
echo Starting MetaAI Pro API Server...
pip install -r requirements.txt
python main.py
pause
'''
        with open(os.path.join(package_dir, "run.bat"), "w", encoding="utf-8") as f:
            f.write(bat_content)
        
        # Linux/Mac shell script
        sh_content = '''#!/bin/bash
echo "Starting MetaAI Pro API Server..."
pip install -r requirements.txt
python main.py
'''
        with open(os.path.join(package_dir, "run.sh"), "w", encoding="utf-8") as f:
            f.write(sh_content)


def create_production_export(model: Any,
                            feature_columns: List[str],
                            target_column: str,
                            task_type: str = "classification",
                            model_name: str = "metaai_model",
                            accuracy: float = 0.0,
                            preprocessors: Dict = None,
                            pydantic_schema_code: Optional[str] = None) -> str:
    """
    Convenience function to create production export package.
    
    Returns path to downloadable zip file.
    """
    exporter = ProductionExporter()
    return exporter.export_production_package(
        model=model,
        feature_columns=feature_columns,
        target_column=target_column,
        task_type=task_type,
        model_name=model_name,
        accuracy=accuracy,
        preprocessors=preprocessors,
        pydantic_schema_code=pydantic_schema_code
    )
