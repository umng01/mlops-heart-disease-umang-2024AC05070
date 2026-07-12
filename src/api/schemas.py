"""
Pydantic models for the Heart Disease Prediction API.

This module defines request and response models with proper validation,
constraints, and documentation for all API endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator, field_validator
from enum import Enum


class SexEnum(int, Enum):
    """Sex enumeration."""
    FEMALE = 0
    MALE = 1


class ChestPainType(int, Enum):
    """Chest pain type enumeration."""
    TYPICAL_ANGINA = 0
    ATYPICAL_ANGINA = 1
    NON_ANGINAL_PAIN = 2
    ASYMPTOMATIC = 3


class RestingECG(int, Enum):
    """Resting electrocardiographic results."""
    NORMAL = 0
    ST_T_ABNORMALITY = 1
    LV_HYPERTROPHY = 2


class STSlope(int, Enum):
    """Slope of the peak exercise ST segment."""
    UPSLOPING = 0
    FLAT = 1
    DOWNSLOPING = 2


class Thalassemia(int, Enum):
    """Thalassemia type."""
    NORMAL = 0
    FIXED_DEFECT = 1
    REVERSIBLE_DEFECT = 2
    UNKNOWN = 3


class PredictionRequest(BaseModel):
    """
    Request model for heart disease prediction.

    Contains all 13 clinical features used by the Cleveland Heart Disease dataset
    for predicting the presence of heart disease.
    """

    age: int = Field(
        ...,
        ge=0,
        le=120,
        description="Age of the patient in years",
        example=63
    )

    sex: int = Field(
        ...,
        ge=0,
        le=1,
        description="Sex of the patient (0 = female, 1 = male)",
        example=1
    )

    cp: int = Field(
        ...,
        ge=0,
        le=3,
        description="Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)",
        example=3
    )

    trestbps: int = Field(
        ...,
        ge=80,
        le=200,
        description="Resting blood pressure in mm Hg on admission to the hospital",
        example=145
    )

    chol: int = Field(
        ...,
        ge=100,
        le=600,
        description="Serum cholesterol in mg/dl",
        example=233
    )

    fbs: int = Field(
        ...,
        ge=0,
        le=1,
        description="Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)",
        example=1
    )

    restecg: int = Field(
        ...,
        ge=0,
        le=2,
        description="Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)",
        example=0
    )

    thalach: int = Field(
        ...,
        ge=60,
        le=220,
        description="Maximum heart rate achieved during exercise",
        example=150
    )

    exang: int = Field(
        ...,
        ge=0,
        le=1,
        description="Exercise induced angina (0 = no, 1 = yes)",
        example=0
    )

    oldpeak: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="ST depression induced by exercise relative to rest",
        example=2.3
    )

    slope: int = Field(
        ...,
        ge=0,
        le=2,
        description="Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)",
        example=0
    )

    ca: int = Field(
        ...,
        ge=0,
        le=4,
        description="Number of major vessels (0-4) colored by fluoroscopy",
        example=0
    )

    thal: int = Field(
        ...,
        ge=0,
        le=3,
        description="Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect, 3 = unknown)",
        example=1
    )

    @field_validator('*', mode='before')
    @classmethod
    def validate_not_none(cls, v, info):
        """Ensure no field is None."""
        if v is None:
            raise ValueError(f"Field '{info.field_name}' cannot be None")
        return v

    @field_validator('trestbps')
    @classmethod
    def validate_blood_pressure(cls, v):
        """Validate blood pressure is in a reasonable range."""
        if v < 80:
            raise ValueError("Resting blood pressure too low (< 80 mm Hg). Please verify the measurement.")
        if v > 200:
            raise ValueError("Resting blood pressure too high (> 200 mm Hg). Please verify the measurement.")
        return v

    @field_validator('chol')
    @classmethod
    def validate_cholesterol(cls, v):
        """Validate cholesterol is in a reasonable range."""
        if v < 100:
            raise ValueError("Cholesterol level too low (< 100 mg/dl). Please verify the measurement.")
        if v > 600:
            raise ValueError("Cholesterol level too high (> 600 mg/dl). Please verify the measurement.")
        return v

    @field_validator('thalach')
    @classmethod
    def validate_heart_rate(cls, v):
        """Validate heart rate is in a reasonable range."""
        if v < 60:
            raise ValueError("Maximum heart rate too low (< 60 bpm). Please verify the measurement.")
        if v > 220:
            raise ValueError("Maximum heart rate too high (> 220 bpm). Please verify the measurement.")
        return v

    @field_validator('oldpeak')
    @classmethod
    def validate_oldpeak(cls, v):
        """Validate ST depression value."""
        if v < 0:
            raise ValueError("ST depression (oldpeak) cannot be negative.")
        if v > 10:
            raise ValueError("ST depression (oldpeak) value too high (> 10). Please verify the measurement.")
        return v

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
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


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class PredictionResponse(BaseModel):
    """
    Response model for heart disease prediction.

    Contains the prediction result, probability score, and risk assessment.
    """

    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Predicted class: 0 = no heart disease, 1 = heart disease present",
        example=1
    )

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of heart disease being present (0.0 to 1.0)",
        example=0.78
    )

    risk_level: str = Field(
        ...,
        description="Risk level assessment: 'low' (< 30%), 'moderate' (30-70%), or 'high' (> 70%)",
        example="high"
    )

    @field_validator('risk_level')
    @classmethod
    def validate_risk_level(cls, v):
        """Ensure risk level is valid."""
        valid_levels = {'low', 'moderate', 'high'}
        if v.lower() not in valid_levels:
            raise ValueError(f"Risk level must be one of {valid_levels}")
        return v.lower()

    @field_validator('probability')
    @classmethod
    def round_probability(cls, v):
        """Round probability to 4 decimal places."""
        return round(v, 4)

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "risk_level": "high"
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    Provides information about the API service status.
    """

    status: str = Field(
        ...,
        description="Service health status",
        example="healthy"
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready for predictions",
        example=True
    )

    version: str = Field(
        ...,
        description="API version",
        example="1.0.0"
    )

    model_path: Optional[str] = Field(
        None,
        description="Path to the loaded model file",
        example="/path/to/model.pkl"
    )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "model_path": "/app/models/model.pkl"
            }
        }


class ErrorResponse(BaseModel):
    """
    Response model for error responses.

    Provides detailed error information for failed requests.
    """

    detail: str = Field(
        ...,
        description="Detailed error message",
        example="Model not loaded. Please try again later."
    )

    status_code: Optional[int] = Field(
        None,
        description="HTTP status code",
        example=503
    )

    error_type: Optional[str] = Field(
        None,
        description="Type of error",
        example="ServiceUnavailable"
    )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "detail": "Model not loaded. Please try again later.",
                "status_code": 503,
                "error_type": "ServiceUnavailable"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions.

    Allows multiple patient records to be predicted in a single request.
    """

    patients: list[PredictionRequest] = Field(
        ...,
        description="List of patient records to predict",
        min_length=1,
        max_length=100
    )

    @field_validator('patients')
    @classmethod
    def validate_batch_size(cls, v):
        """Validate batch size limits."""
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 patients")
        if len(v) < 1:
            raise ValueError("Batch must contain at least 1 patient")
        return v

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "patients": [
                    {
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
                    },
                    {
                        "age": 52,
                        "sex": 0,
                        "cp": 2,
                        "trestbps": 125,
                        "chol": 212,
                        "fbs": 0,
                        "restecg": 1,
                        "thalach": 168,
                        "exang": 0,
                        "oldpeak": 1.0,
                        "slope": 1,
                        "ca": 2,
                        "thal": 2
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions.

    Contains predictions for all patients in the batch.
    """

    predictions: list[PredictionResponse] = Field(
        ...,
        description="List of prediction results for each patient"
    )

    total_count: int = Field(
        ...,
        description="Total number of predictions made",
        example=2
    )

    high_risk_count: int = Field(
        ...,
        description="Number of high-risk patients in the batch",
        example=1
    )

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 1,
                        "probability": 0.78,
                        "risk_level": "high"
                    },
                    {
                        "prediction": 0,
                        "probability": 0.23,
                        "risk_level": "low"
                    }
                ],
                "total_count": 2,
                "high_risk_count": 1
            }
        }
