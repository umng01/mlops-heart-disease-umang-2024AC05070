# MLOps Heart Disease Prediction Pipeline

**Course**: MLOps - BITS Pilani  
**Author**: Umang Sharma  
**Roll Number**: 2024AC05070  
**Date**: July 12, 2026

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.7.1-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## 🎯 Project Overview

Complete end-to-end MLOps pipeline for heart disease prediction using UCI Heart Disease dataset. This project demonstrates industry-standard ML engineering practices including experiment tracking, model deployment, containerization, and monitoring.

## 📊 Key Results

- **Best Model**: Logistic Regression & Random Forest
- **Test ROC-AUC**: **95.78%**
- **Test Accuracy**: 86.89% / 88.52%
- **Models Trained**: 3 (Logistic Regression, Random Forest, XGBoost)
- **Total Experiments**: Tracked with MLflow

## 🏗️ Architecture

```
Data → EDA → Preprocessing → Model Training → Evaluation → Deployment
         ↓                        ↓              ↓            ↓
    Visualizations         MLflow Tracking   FastAPI      Docker
                                                           Kubernetes
```

## 📁 Project Structure

```
mlops-heart-disease-project/
├── data/
│   ├── raw/                    # Original UCI dataset
│   └── processed/              # Train/test splits
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── api/                    # FastAPI application
│   ├── data/                   # Data processing
│   ├── models/                 # Model training & prediction
│   └── utils/                  # Configuration
├── models/                     # Saved models
├── reports/
│   ├── figures/               # Generated plots
│   └── MLOps_Assignment_Report_Umang_Sharma.md
├── tests/                     # Unit & integration tests
├── deployment/
│   └── kubernetes/            # K8s manifests
├── monitoring/                # Prometheus config
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/umng01/mlops-heart-disease-umang-2024AC05070.git
cd mlops-heart-disease-umang-2024AC05070

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_eda_preprocessing.ipynb
# 2. notebooks/02_model_training.ipynb
```

### 3. View MLflow Experiments

```bash
mlflow ui
# Open http://localhost:5000
```

### 4. Start API Server

```bash
uvicorn src.api.app:app --reload
# API docs at http://localhost:8000/docs
```

### 5. Test Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @data/sample_input.json
```

## 🐳 Docker Deployment

### Build & Run

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8000:8000 heart-disease-api:latest

# Or use Docker Compose
docker-compose up -d
```

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

# Check status
kubectl get pods
kubectl get services
```

## 📈 Dataset Information

**Source**: UCI Machine Learning Repository  
**Dataset**: Heart Disease (Cleveland)  
**Instances**: 303 patients  
**Features**: 13 clinical attributes  
**Target**: Binary (Disease / No Disease)

### Features

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Gender (1=male, 0=female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia (1-3) |

## 🤖 Models Trained

### 1. Logistic Regression
- **Test ROC-AUC**: 95.78%
- **Test Accuracy**: 86.89%
- **Hyperparameters**: C=0.1, penalty='l2', solver='liblinear'

### 2. Random Forest
- **Test ROC-AUC**: 95.78%
- **Test Accuracy**: 88.52%
- **Hyperparameters**: n_estimators=50, max_depth=10

### 3. XGBoost
- **Test ROC-AUC**: 94.16%
- **Test Accuracy**: 86.89%
- **Hyperparameters**: n_estimators=50, max_depth=5, learning_rate=0.05

## 📊 Model Comparison

![Model Comparison](reports/figures/model_comparison.png)

![ROC Curves](reports/figures/roc_curves_comparison.png)

## 🔍 Feature Importance

![Feature Importance](reports/figures/feature_importance.png)

**Top 5 Features**:
1. Number of major vessels (ca)
2. Thalassemia type (thal)
3. ST depression (oldpeak)
4. Maximum heart rate (thalach)
5. Chest pain type (cp)

## 📝 MLflow Tracking

All experiments tracked with:
- **Parameters**: Hyperparameters for each model
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Artifacts**: Models, plots, classification reports
- **Tags**: Author, model type, framework

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_api.py -v
```

## 📊 Monitoring

### Prometheus Metrics
- API request rate & latency
- Prediction count by class
- Model inference time
- Error rates

### Grafana Dashboards
Access at http://localhost:3000 (default: admin/admin)

## 🔄 CI/CD Pipeline

GitHub Actions workflow includes:
1. **Linting**: Code quality checks with flake8
2. **Testing**: Unit & integration tests with pytest
3. **Building**: Docker image creation
4. **Deployment**: Push to registry & K8s update

## 📚 Documentation

- **Full Report**: [reports/MLOps_Assignment_Report_Umang_Sharma.md](reports/MLOps_Assignment_Report_Umang_Sharma.md)
- **API Docs**: http://localhost:8000/docs (when server running)
- **Setup Guide**: [docs/SETUP.md](docs/SETUP.md)

## 🎥 Demo Video

[Link to demo video - YouTube/Google Drive]

## 📄 License

This project is for educational purposes as part of BITS Pilani MLOps course.

## 👤 Author

**Umang Sharma**  
Roll Number: 2024AC05070  
BITS Pilani  

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Cleveland Clinic Foundation for data collection
- BITS Pilani for the MLOps course

---

**Last Updated**: July 12, 2026
