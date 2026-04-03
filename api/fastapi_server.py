"""
FastAPI Production Server
Integrates: FastAPI, Pydantic for REST API deployment
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import io
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_training import train_model
from core.explainability import ExplainabilityEngine
from core.agentic_researcher import AgenticResearchOrchestrator

# ==========================================
# FASTAPI APP INITIALIZATION
# ==========================================

app = FastAPI(
    title="Meta AI Builder API",
    description="Production-grade AutoML API with explainability and visualization",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PYDANTIC MODELS
# ==========================================

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[List[float]] = Field(..., description="Feature matrix as list of lists")
    model_name: str = Field("RandomForest", description="Model to use for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [[1.0, 2.0, 3.0, 4.0]],
                "model_name": "RandomForest"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    model_used: str
    timestamp: str


class TrainingRequest(BaseModel):
    """Request model for training"""
    target_column: str = Field(..., description="Name of target column")
    model_names: List[str] = Field(["RandomForest", "XGBoost"], description="Models to train")
    optimize: bool = Field(False, description="Enable AutoML optimization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_column": "target",
                "model_names": ["RandomForest", "XGBoost", "LightGBM"],
                "optimize": True
            }
        }


class TrainingResponse(BaseModel):
    """Response model for training"""
    status: str
    models_trained: List[str]
    metrics: Dict[str, Dict[str, float]]
    best_model: str
    timestamp: str


class ExplainRequest(BaseModel):
    """Request model for explanations"""
    features: List[List[float]]
    model_name: str = "RandomForest"
    method: str = Field("shap", description="Explanation method: 'shap' or 'lime'")
    instance_idx: int = Field(0, description="Instance index to explain")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    models_available: List[str]


class ResearchRequest(BaseModel):
    """Request model for agentic research"""
    target_column: str
    target_metric: str = "f1"
    threshold: float = 0.85
    max_iterations: int = 3
    dataset_name: str = "default"


# ==========================================
# GLOBAL STATE (In production, use Redis/DB)
# ==========================================

class ModelRegistry:
    """Simple in-memory model registry"""
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self.datasets: Dict[str, pd.DataFrame] = {}
    
    def register_model(self, name: str, model: Any, metadata: Dict):
        """Register a trained model"""
        self.models[name] = model
        self.metadata[name] = metadata
    
    def get_model(self, name: str):
        """Retrieve a model"""
        if name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return self.models[name]
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())
    
    def store_dataset(self, name: str, df: pd.DataFrame):
        """Store dataset"""
        self.datasets[name] = df
    
    def get_dataset(self, name: str) -> pd.DataFrame:
        """Retrieve dataset"""
        if name not in self.datasets:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
        return self.datasets[name]


# Initialize registry
registry = ModelRegistry()

# ==========================================
# AUTHENTICATION (Simple API Key)
# ==========================================

async def verify_api_key(x_api_key: str = Header(None)):
    """Simple API key verification"""
    # In production, use proper authentication
    VALID_API_KEYS = os.getenv("API_KEYS", "dev-key-123").split(",")
    
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return x_api_key


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <html>
        <head>
            <title>Meta AI Builder API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #3b82f6; }
                .endpoint { background: #f3f4f6; padding: 10px; margin: 10px 0; border-radius: 5px; }
                code { background: #1f2937; color: #10b981; padding: 2px 6px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🧬 Meta AI Builder API v3.0</h1>
            <p>Production-grade AutoML API with explainability and visualization</p>
            
            <h2>📚 Documentation</h2>
            <div class="endpoint">
                <strong>Interactive API Docs:</strong> <a href="/docs">/docs</a>
            </div>
            <div class="endpoint">
                <strong>ReDoc:</strong> <a href="/redoc">/redoc</a>
            </div>
            
            <h2>🔑 Key Endpoints</h2>
            <div class="endpoint">
                <code>POST /predict</code> - Make predictions
            </div>
            <div class="endpoint">
                <code>POST /train</code> - Train models
            </div>
            <div class="endpoint">
                <code>POST /explain</code> - Get SHAP/LIME explanations
            </div>
            <div class="endpoint">
                <code>GET /models</code> - List available models
            </div>
            <div class="endpoint">
                <code>GET /health</code> - Health check
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        timestamp=datetime.now().isoformat(),
        models_available=registry.list_models()
    )


@app.get("/models")
async def list_models():
    """List all available models"""
    models = registry.list_models()
    metadata = {name: registry.metadata.get(name, {}) for name in models}
    
    return {
        "models": models,
        "count": len(models),
        "metadata": metadata
    }


@app.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = "default"
):
    """Upload a CSV dataset"""
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Store dataset
        registry.store_dataset(dataset_name, df)
        
        # Profile dataset
        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "profile": profile
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload dataset: {str(e)}")


@app.post("/train", response_model=TrainingResponse)
async def train_models(
    request: TrainingRequest,
    dataset_name: str = "default"
):
    """Train models on uploaded dataset"""
    try:
        # Get dataset
        df = registry.get_dataset(dataset_name)
        
        # Validate target column
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{request.target_column}' not found in dataset"
            )
        
        # Train models
        results = {}
        trained_models = []
        
        for model_name in request.model_names:
            try:
                model, metrics = train_model(
                    model_name,
                    df,
                    request.target_column,
                    optimize=request.optimize
                )
                
                if model:
                    # Register model
                    registry.register_model(
                        model_name,
                        model,
                        {
                            "trained_at": datetime.now().isoformat(),
                            "dataset": dataset_name,
                            "target_column": request.target_column,
                            "metrics": metrics
                        }
                    )
                    
                    results[model_name] = metrics
                    trained_models.append(model_name)
                    
            except Exception as e:
                print(f"Failed to train {model_name}: {e}")
                continue
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 0))[0] if results else None
        
        return TrainingResponse(
            status="success",
            models_trained=trained_models,
            metrics=results,
            best_model=best_model,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using a trained model"""
    try:
        # Get model
        model = registry.get_model(request.model_name)
        
        # Convert features to numpy array
        X = np.array(request.features)
        
        # Make predictions
        predictions = model.predict(X).tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X).tolist()
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_used=request.model_name,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """Generate SHAP or LIME explanations"""
    try:
        # Get model
        model = registry.get_model(request.model_name)
        
        # Convert features to DataFrame
        X = pd.DataFrame(request.features)
        
        # Create explainer
        explainer = ExplainabilityEngine()
        
        if request.method.lower() == 'shap':
            result = explainer.explain_with_shap(model, X)
        elif request.method.lower() == 'lime':
            result = explainer.explain_with_lime(model, X, request.instance_idx)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        # Convert numpy arrays to lists for JSON serialization
        if 'shap_values' in result:
            if isinstance(result['shap_values'], list):
                result['shap_values'] = [v.tolist() if isinstance(v, np.ndarray) else v for v in result['shap_values']]
            elif isinstance(result['shap_values'], np.ndarray):
                result['shap_values'] = result['shap_values'].tolist()
        
        # Remove non-serializable objects
        result.pop('explainer', None)
        result.pop('X', None)
        result.pop('explanation', None)
        
        return {
            "status": "success",
            "method": request.method,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/research")
async def run_agentic_research(request: ResearchRequest):
    """Trigger the Autonomous Agentic Research Lab"""
    try:
        df = registry.get_dataset(request.dataset_name)
        
        orchestrator = AgenticResearchOrchestrator(
            target_metric=request.target_metric, 
            threshold=request.threshold
        )
        
        results = orchestrator.run_research_cycle(
            df, 
            request.target_column, 
            max_iterations=request.max_iterations
        )
        
        # Register the final best model
        if "model" in results:
            registry.register_model(
                f"Agentic_{request.target_column}",
                results["model"],
                {
                    "trained_at": datetime.now().isoformat(),
                    "type": "agentic_optimized",
                    "metrics": results["metrics"]
                }
            )
            
        # Remove non-serializable objects for JSON response
        results.pop("model", None)
        
        return {
            "status": "success",
            "research_results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research cycle failed: {str(e)}")


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ==========================================
# STARTUP/SHUTDOWN EVENTS
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Meta AI Builder API starting up...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Meta AI Builder API shutting down...")


# ==========================================
# MAIN (for direct execution)
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "fastapi_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
