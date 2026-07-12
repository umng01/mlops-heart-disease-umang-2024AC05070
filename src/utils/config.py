"""
Configuration Management
Author: Umang Sharma (2024AC05070)
"""

import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT_NAME = "heart-disease-prediction"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature Names
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

TARGET_NAME = 'target_binary'

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Heart Disease Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Heart Disease Prediction API using Machine Learning

**Student:** Umang Sharma
**Roll No:** 2024AC05070
**Course:** AIMLCZG523 - MLOps

## Features
- Predict heart disease risk
- Return probability scores
- Health monitoring endpoints
"""

# Model Hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [1000]
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

XGBOOST_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
