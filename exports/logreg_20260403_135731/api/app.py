"""
MetaAI Pro - Production API Server
Auto-generated deployment endpoint
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import pandas as pd
from pathlib import Path
import json

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
metadata = {}


@app.on_event("startup")
async def load_model():
    global model, metadata
    
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded: {type(model).__name__}")
    
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        print(f"Metadata loaded: {metadata.get('model_name')}")


# Request schema
class PredictionRequest(BaseModel):
    age: float
    bmi: float
    income: float


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
FEATURE_MAPPING = {
    "age": "age",
    "bmi": "bmi",
    "income": "income"
}
FEATURE_ORDER = ["age", "bmi", "income"]


def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    """Convert request to DataFrame with correct feature order."""
    data = request.dict()
    features = {}
    
    for api_name, orig_name in FEATURE_MAPPING.items():
        if api_name in data:
            features[orig_name] = data[api_name]
    
    return pd.DataFrame([features])[FEATURE_ORDER]


@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint."""
    return {
        "name": "MetaAI Pro Model API",
        "version": "1.0.0",
        "status": "running"
    }


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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = prepare_features(request)
        prediction = float(model.predict(df)[0])
        
        probability = None
        confidence = None
        label = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
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
            pred = float(model.predict(df)[0])
            
            prob = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)[0]
                prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            
            results.append({"prediction": pred, "probability": prob, "status": "success"})
        except Exception as e:
            results.append({"status": "error", "detail": str(e)})
    
    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
