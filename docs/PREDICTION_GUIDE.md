# Heart Disease Prediction Guide

Comprehensive guide for using the heart disease prediction module.

**Author:** Umang Sharma (2024AC05070)

## Overview

The prediction module provides multiple interfaces for making heart disease predictions:
- **Programmatic API** - Import and use in Python code
- **Command Line Interface** - Run predictions from terminal
- **File-based Input** - Process JSON or CSV files
- **API Integration** - Ready for FastAPI integration

## Quick Start

### 1. Train a Model First

```bash
# Train models (required before predictions)
python src/models/train.py
```

### 2. Make Your First Prediction

```bash
# View feature information
python src/models/predict.py --features

# Predict from JSON file
python src/models/predict.py --input data/sample_input.json

# Predict with CLI arguments
python src/models/predict.py --age 63 --sex 1 --cp 3 --trestbps 145 \
  --chol 233 --fbs 1 --restecg 0 --thalach 150 --exang 0 \
  --oldpeak 2.3 --slope 0 --ca 0 --thal 1
```

## Feature Descriptions

| Feature | Description | Range |
|---------|-------------|-------|
| age | Age in years | 29-77 |
| sex | Sex (1=male, 0=female) | 0-1 |
| cp | Chest pain type | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | 94-200 |
| chol | Serum cholesterol (mg/dl) | 126-564 |
| fbs | Fasting blood sugar > 120 mg/dl | 0-1 |
| restecg | Resting ECG results | 0-2 |
| thalach | Maximum heart rate achieved | 71-202 |
| exang | Exercise induced angina | 0-1 |
| oldpeak | ST depression | 0.0-6.2 |
| slope | Slope of peak exercise ST segment | 0-2 |
| ca | Number of major vessels | 0-4 |
| thal | Thalassemia | 0-3 |

## Usage Methods

### Method 1: Command Line Interface

#### Show Feature Information
```bash
python src/models/predict.py --features
```

#### Single Prediction from JSON
```bash
python src/models/predict.py --input data/sample_input.json
```

#### Batch Prediction from CSV
```bash
python src/models/predict.py --input data/sample_batch.csv --output predictions.csv
```

#### Direct CLI Input
```bash
python src/models/predict.py \
  --age 63 --sex 1 --cp 3 --trestbps 145 --chol 233 \
  --fbs 1 --restecg 0 --thalach 150 --exang 0 \
  --oldpeak 2.3 --slope 0 --ca 0 --thal 1
```

#### Specify Model Path
```bash
python src/models/predict.py \
  --model-path models/random_forest_pipeline.pkl \
  --input data/sample_input.json
```

### Method 2: Python API

#### Single Prediction

```python
from src.models.predict import HeartDiseasePredictor

# Initialize predictor (auto-loads best model)
predictor = HeartDiseasePredictor()

# Patient data
patient = {
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
    'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}

# Make prediction
result = predictor.predict_single(patient)

print(f"Prediction: {result['prediction_label']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.4f}")
```

#### Batch Prediction

```python
import pandas as pd
from src.models.predict import HeartDiseasePredictor

# Initialize predictor
predictor = HeartDiseasePredictor()

# Load data
data = pd.read_csv('patients.csv')

# Make predictions
results = predictor.predict_batch(data)

# View results
print(results[['prediction_label', 'probability_positive', 'risk_level']])

# Save results
results.to_csv('predictions.csv', index=False)
```

#### Raw Predictions

```python
from src.models.predict import HeartDiseasePredictor

predictor = HeartDiseasePredictor()

# Get raw predictions and probabilities
predictions, probabilities = predictor.predict(data, return_proba=True)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

### Method 3: API Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.predict import HeartDiseasePredictor

# Initialize predictor once at startup
predictor = HeartDiseasePredictor()

app = FastAPI()

class PredictionRequest(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert to dict
        data = request.dict()
        
        # Make prediction
        result = predictor.predict_single(data)
        
        return {
            "success": True,
            "prediction": result['prediction_label'],
            "risk_level": result['risk_level'],
            "confidence": result['confidence'],
            "probabilities": {
                "no_disease": result['probability_negative'],
                "disease": result['probability_positive']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## Input Formats

### JSON Format (Single Prediction)

```json
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
}
```

### CSV Format (Batch Prediction)

```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
63,1,3,145,233,1,0,150,0,2.3,0,0,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2
41,0,1,130,204,0,0,172,0,1.4,2,0,2
```

## Output Formats

### Single Prediction Output

```python
{
    'prediction': 1,
    'prediction_label': 'Heart Disease',
    'probability_negative': 0.2345,
    'probability_positive': 0.7655,
    'risk_level': 'High',
    'confidence': 0.7655,
    'input_features': {...}
}
```

### Batch Prediction Output (DataFrame)

| age | sex | cp | ... | prediction_label | probability_positive | risk_level | confidence |
|-----|-----|----|----|------------------|---------------------|------------|------------|
| 63  | 1   | 3  | ... | Heart Disease    | 0.7655              | High       | 0.7655     |
| 37  | 1   | 2  | ... | No Heart Disease | 0.3421              | Moderate   | 0.6579     |

## Risk Level Classification

- **Low Risk**: Probability < 0.3
- **Moderate Risk**: Probability 0.3 - 0.6
- **High Risk**: Probability > 0.6

## Input Validation

The predictor automatically validates inputs:

```python
from src.models.predict import HeartDiseasePredictor

predictor = HeartDiseasePredictor()

# Check if input is valid
is_valid, errors = predictor.validate_input(data)

if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

Validation checks:
- All required features present
- No null/missing values
- Values within valid ranges
- Correct data types

## Model Selection

The predictor automatically selects the best model based on ROC-AUC score from training.

To use a specific model:

```python
# Python API
predictor = HeartDiseasePredictor(model_path='models/random_forest_pipeline.pkl')

# CLI
python src/models/predict.py --model-path models/xgboost_pipeline.pkl --input data.json
```

## Error Handling

```python
from src.models.predict import HeartDiseasePredictor

predictor = HeartDiseasePredictor()

try:
    result = predictor.predict_single(patient_data)
except ValueError as e:
    print(f"Validation error: {e}")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
except Exception as e:
    print(f"Prediction error: {e}")
```

## Performance Tips

### 1. Reuse Predictor Instance

```python
# GOOD - Initialize once
predictor = HeartDiseasePredictor()
for patient in patients:
    result = predictor.predict_single(patient)

# BAD - Initialize multiple times
for patient in patients:
    predictor = HeartDiseasePredictor()  # Slow!
    result = predictor.predict_single(patient)
```

### 2. Use Batch Prediction

```python
# GOOD - Batch processing
results = predictor.predict_batch(patients_df)

# LESS EFFICIENT - Loop
results = []
for _, patient in patients_df.iterrows():
    result = predictor.predict_single(patient.to_dict())
    results.append(result)
```

### 3. Skip Validation for Trusted Data

```python
# Skip validation for pre-validated data
result = predictor.predict_single(data, validate=False)
```

## Examples

Run comprehensive examples:

```bash
python examples/prediction_examples.py
```

This includes:
1. Single prediction
2. Batch prediction
3. Raw predictions
4. Input validation
5. Feature information
6. Model information
7. API integration pattern
8. File-based prediction

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: No trained models found in models
```

**Solution:** Train a model first
```bash
python src/models/train.py
```

### Validation Error

```
ValueError: Feature 'age' has values outside valid range [29, 77]: [120]
```

**Solution:** Check input values match expected ranges
```bash
python src/models/predict.py --features
```

### Missing Features Error

```
ValueError: Missing required features: ['ca', 'thal']
```

**Solution:** Ensure all 13 features are provided

## Integration with Other Components

### With API Server
```python
# src/api/main.py
from src.models.predict import HeartDiseasePredictor

# Initialize at startup
predictor = HeartDiseasePredictor()

@app.post("/predict")
async def predict(request: PredictionRequest):
    return predictor.predict_single(request.dict())
```

### With Data Pipeline
```python
# After preprocessing
from src.models.predict import HeartDiseasePredictor

predictor = HeartDiseasePredictor()
predictions = predictor.predict_batch(processed_data)
```

### With Monitoring
```python
from src.models.predict import HeartDiseasePredictor
import mlflow

predictor = HeartDiseasePredictor()

with mlflow.start_run():
    result = predictor.predict_single(data)
    mlflow.log_metric("prediction", result['prediction'])
    mlflow.log_metric("confidence", result['confidence'])
```

## Best Practices

1. **Always validate** inputs before prediction
2. **Reuse predictor** instances for better performance
3. **Use batch prediction** for multiple samples
4. **Handle errors** gracefully in production
5. **Log predictions** for monitoring and auditing
6. **Version models** to track which model made predictions
7. **Monitor confidence** scores to detect model drift

## Testing

Run tests:
```bash
pytest tests/test_predict.py -v
```

Create test cases:
```python
def test_prediction():
    predictor = HeartDiseasePredictor()
    
    sample = {...}  # Valid sample
    result = predictor.predict_single(sample)
    
    assert 'prediction' in result
    assert 'risk_level' in result
    assert 0 <= result['confidence'] <= 1
```

## Further Reading

- [Model Training Guide](../README.md#model-training)
- [API Documentation](../README.md#api-endpoints)
- [Feature Engineering](../README.md#data-preprocessing)
- [MLflow Tracking](../README.md#mlflow-tracking)

---

For questions or issues, contact: Umang Sharma (2024AC05070)
