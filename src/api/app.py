"""
FastAPI application for heart disease prediction.

This module provides REST API endpoints for:
- Health checks
- Heart disease prediction with probability scores
- Prometheus metrics
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, validator
from starlette.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['prediction']
)
MODEL_LOAD_TIME = Histogram(
    'model_load_duration_seconds',
    'Model loading duration in seconds'
)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using machine learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODEL = None
MODEL_PATH = None


class PatientFeatures(BaseModel):
    """Request model for patient features."""

    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0=normal, 1=fixed defect, 2=reversible defect, 3=unknown)")

    @validator('*', pre=True)
    def validate_numeric(cls, v):
        """Ensure all values are numeric."""
        if v is None:
            raise ValueError("Value cannot be None")
        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    prediction: int = Field(..., description="Predicted class (0=no disease, 1=disease)")
    probability: float = Field(..., ge=0, le=1, description="Probability of having heart disease")
    risk_level: str = Field(..., description="Risk level (low, moderate, high)")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "risk_level": "high"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


def get_risk_level(probability: float) -> str:
    """
    Determine risk level based on probability.

    Args:
        probability: Probability of heart disease (0-1)

    Returns:
        Risk level string
    """
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "moderate"
    else:
        return "high"


def load_model(model_path: str = None) -> Any:
    """
    Load the trained model from disk.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded model object

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    global MODEL, MODEL_PATH

    if model_path is None:
        # Default model path
        model_path = Path(__file__).parent.parent.parent / "models" / "model.pkl"

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"Loading model from {model_path}")
    start_time = time.time()

    try:
        MODEL = joblib.load(model_path)
        MODEL_PATH = str(model_path)
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observe(load_time)
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        return MODEL
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        logger.warning("API will start but predictions will fail until model is loaded")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log requests and track metrics.

    Args:
        request: The incoming request
        call_next: The next middleware/endpoint

    Returns:
        Response from the endpoint
    """
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    # Log response
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"Status={response.status_code} Duration={duration:.3f}s"
    )

    return response


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        version="1.0.0"
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: PatientFeatures) -> PredictionResponse:
    """
    Predict heart disease based on patient features.

    Args:
        features: Patient features

    Returns:
        Prediction result with probability and risk level

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if MODEL is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        # Convert features to DataFrame
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])

        # Ensure correct column order
        expected_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        df = df[expected_columns]

        logger.info(f"Making prediction for features: {feature_dict}")

        # Make prediction
        prediction = int(MODEL.predict(df)[0])

        # Get probability (handle different model types)
        if hasattr(MODEL, 'predict_proba'):
            probability = float(MODEL.predict_proba(df)[0][1])
        else:
            # For models without predict_proba, use decision_function or default
            if hasattr(MODEL, 'decision_function'):
                decision = MODEL.decision_function(df)[0]
                # Convert to probability using sigmoid
                probability = float(1 / (1 + np.exp(-decision)))
            else:
                probability = float(prediction)

        # Determine risk level
        risk_level = get_risk_level(probability)

        # Update metrics
        PREDICTION_COUNT.labels(prediction=prediction).inc()

        logger.info(
            f"Prediction complete: class={prediction}, "
            f"probability={probability:.3f}, risk={risk_level}"
        )

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """
    Handle ValueError exceptions.

    Args:
        request: The request that caused the error
        exc: The exception

    Returns:
        JSON error response
    """
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.

    Args:
        request: The request that caused the error
        exc: The exception

    Returns:
        JSON error response
    """
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    # Load model before starting server
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
