"""
Production FastAPI application with full middleware stack.
"""
from contextlib import asynccontextmanager
from time import perf_counter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.utils.logger import get_logger
from app.utils.telemetry import telemetry
from app.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, clean up on shutdown."""
    logger.info("META-AI API starting up...")
    # Add any startup logic here (e.g. warming up providers)
    yield
    logger.info("META-AI API shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade AI platform powered by Meta AI with RAG",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ── Middleware Stack ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routes ────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    latency_ms = (perf_counter() - started) * 1000.0
    telemetry.record(
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms
    )
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    return response


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}


@app.get("/metrics")
async def metrics():
    return telemetry.snapshot()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
