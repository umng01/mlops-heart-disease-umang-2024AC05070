# Prediction Quick Reference

Fast reference guide for making predictions with the heart disease model.

## Installation & Setup

```bash
# Train model (required first)
python src/models/train.py

# Verify model exists
ls -la models/
```

## Command Line Usage

```bash
# Show features
python src/models/predict.py --features

# JSON input
python src/models/predict.py --input data/sample_input.json

# CSV batch
python src/models/predict.py --input data.csv --output results.csv

# CLI args
python src/models/predict.py --age 63 --sex 1 --cp 3 --trestbps 145 \
  --chol 233 --fbs 1 --restecg 0 --thalach 150 --exang 0 \
  --oldpeak 2.3 --slope 0 --ca 0 --thal 1
```

## Python API

### Single Prediction
```python
from src.models.predict import HeartDiseasePredictor

predictor = HeartDiseasePredictor()

patient = {
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
    'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}

result = predictor.predict_single(patient)
print(f"{result['prediction_label']} - {result['risk_level']}")
```

### Batch Prediction
```python
import pandas as pd
predictor = HeartDiseasePredictor()

data = pd.read_csv('patients.csv')
results = predictor.predict_batch(data)
results.to_csv('predictions.csv', index=False)
```

### Raw Prediction
```python
predictions, probabilities = predictor.predict(data, return_proba=True)
```

## Features (13 required)

| Feature | Type | Range | Example |
|---------|------|-------|---------|
| age | float | 29-77 | 63 |
| sex | int | 0-1 | 1 |
| cp | int | 0-3 | 3 |
| trestbps | float | 94-200 | 145 |
| chol | float | 126-564 | 233 |
| fbs | int | 0-1 | 1 |
| restecg | int | 0-2 | 0 |
| thalach | float | 71-202 | 150 |
| exang | int | 0-1 | 0 |
| oldpeak | float | 0-6.2 | 2.3 |
| slope | int | 0-2 | 0 |
| ca | int | 0-4 | 0 |
| thal | int | 0-3 | 1 |

## Output Format

```python
{
    'prediction': 1,  # 0 or 1
    'prediction_label': 'Heart Disease',  # or 'No Heart Disease'
    'probability_positive': 0.7655,  # 0-1
    'probability_negative': 0.2345,  # 0-1
    'risk_level': 'High',  # Low/Moderate/High
    'confidence': 0.7655  # 0-1
}
```

## Risk Levels
- **Low**: < 30% probability
- **Moderate**: 30-60% probability
- **High**: > 60% probability

## Common Patterns

### API Integration
```python
from fastapi import FastAPI
from src.models.predict import HeartDiseasePredictor

app = FastAPI()
predictor = HeartDiseasePredictor()

@app.post("/predict")
async def predict(data: dict):
    return predictor.predict_single(data)
```

### Error Handling
```python
try:
    result = predictor.predict_single(data)
except ValueError as e:
    print(f"Invalid input: {e}")
except FileNotFoundError:
    print("Model not found - train first")
```

### Validation
```python
is_valid, errors = predictor.validate_input(data)
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

## Testing

```bash
# Run tests
pytest tests/test_predict.py -v

# Run examples
python examples/prediction_examples.py
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| Model not found | Run `python src/models/train.py` |
| Missing features | Check all 13 features present |
| Value out of range | Use `--features` to see valid ranges |
| Invalid file format | Use `.json` or `.csv` only |

## Performance

- ✅ **Reuse predictor** - Initialize once
- ✅ **Use batch** - Process multiple samples together
- ✅ **Skip validation** - For pre-validated data
- ❌ **Don't re-initialize** - Slow model loading

## File Formats

### JSON (Single)
```json
{"age": 63, "sex": 1, "cp": 3, ...}
```

### CSV (Batch)
```csv
age,sex,cp,trestbps,...
63,1,3,145,...
37,1,2,130,...
```

## Advanced

### Custom Model
```python
predictor = HeartDiseasePredictor(model_path='models/custom.pkl')
```

### Skip Validation
```python
result = predictor.predict_single(data, validate=False)
```

### Get Info
```python
predictor.get_model_info()
predictor.get_feature_info()
```

---

**Full Guide:** [PREDICTION_GUIDE.md](PREDICTION_GUIDE.md)
