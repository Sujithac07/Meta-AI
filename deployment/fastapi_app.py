from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Extra
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Meta AI Builder++ Inference API",
    description="High-performance model serving for financial market predictions.",
    version="1.0.0"
)

# --- Models ---
class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_name: Optional[str] = "RandomForest"

    class Config:
        extra = Extra.allow

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    model_used: str

# --- State ---
MODELS_DIR = "models"
loaded_models = {}

def load_all_models():
    if not os.path.exists(MODELS_DIR):
        print(f"Warning: Models directory {MODELS_DIR} not found.")
        return
    
    for model_file in os.listdir(MODELS_DIR):
        if model_file.endswith(".joblib") or model_file.endswith(".pkl"):
            name = model_file.split(".")[0]
            try:
                loaded_models[name] = joblib.load(os.path.join(MODELS_DIR, model_file))
                print(f"Loaded model: {name}")
            except Exception as e:
                print(f"Failed to load {model_file}: {e}")

@app.on_event("startup")
async def startup_event():
    load_all_models()

@app.get("/")
async def root():
    return {
        "status": "online",
        "available_models": list(loaded_models.keys()),
        "message": "Welcome to the Meta AI Builder++ API. Use /predict for inference."
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    model_name = request.model_name
    if model_name not in loaded_models:
        # Try to use any available model if requested is missing
        if not loaded_models:
             raise HTTPException(status_code=503, detail="No models loaded on server.")
        model_name = list(loaded_models.keys())[0]

    model = loaded_models[model_name]
    
    try:
        df = pd.DataFrame(request.data)
        predictions = model.predict(df).tolist()
        
        probabilities = None
        if hasattr(model, "predict_proba"):
             probabilities = model.predict_proba(df).tolist()
             
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_used=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
