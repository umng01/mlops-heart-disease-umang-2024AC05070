# 🚀 Quick Start Guide

**Student:** Umang Sharma | **Roll No:** 2024AC05070

---

## Prerequisites

- Python 3.9+
- Docker
- kubectl (for Kubernetes deployment)
- Git

---

## 🎯 5-Minute Setup

```bash
# 1. Clone/Navigate to project
cd /Users/umang.sharma/mlops-heart-disease-project

# 2. Run automated setup
chmod +x scripts/init_project.sh
./scripts/init_project.sh

# 3. Activate environment
source venv/bin/activate

# 4. Download data
python src/data/download_data.py
```

---

## 📊 Run EDA & Training

```bash
# Option 1: Using Jupyter Notebooks
jupyter notebook notebooks/

# Run: 01_eda_preprocessing.ipynb
# Then: 02_model_training.ipynb

# Option 2: Using Scripts
python src/data/preprocess.py
python src/models/train.py
```

---

## 🔬 View MLflow Experiments

```bash
mlflow ui
# Open: http://localhost:5000
```

---

## 🌐 Run API Locally

```bash
# Start FastAPI server
uvicorn src.api.app:app --reload

# Test in another terminal
./scripts/test_api.sh http://localhost:8000
```

---

## 🐳 Docker Deployment

```bash
# Build image
./scripts/build_docker.sh

# Run container
docker run -p 8000:8000 heart-disease-api:latest

# Test
curl http://localhost:8000/health
```

---

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
./scripts/deploy_k8s.sh

# Check status
kubectl get pods
kubectl get services
```

---

## 🧪 Run Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📁 Project Structure

```
mlops-heart-disease-project/
├── notebooks/          # EDA & Training notebooks
├── src/               # Source code
│   ├── data/         # Data processing
│   ├── models/       # Model training
│   ├── api/          # FastAPI application
│   └── utils/        # Utilities & config
├── tests/            # Unit tests
├── deployment/       # K8s manifests
├── scripts/          # Helper scripts
└── models/           # Saved models
```

---

## 🎥 Demo Video Script

1. **Introduction** (30s)
   - Project overview
   - Problem statement

2. **EDA** (1 min)
   - Run notebook
   - Show visualizations

3. **Model Training** (1.5 min)
   - Train models
   - MLflow UI tour
   - Model comparison

4. **API Demo** (1 min)
   - Start API
   - Make predictions
   - Show Swagger docs

5. **Docker** (1 min)
   - Build image
   - Run container
   - Test API

6. **Kubernetes** (1.5 min)
   - Deploy
   - Show pods/services
   - Access endpoint

7. **Monitoring** (30s)
   - Show logs
   - Metrics

---

## 📝 Submission Checklist

- [ ] All code committed to GitHub
- [ ] Screenshots in `screenshots/` folder
- [ ] MLflow experiments recorded
- [ ] Docker image builds successfully
- [ ] K8s deployment works
- [ ] Tests pass
- [ ] README complete
- [ ] 10-page report ready
- [ ] Video recorded

---

## 💡 Tips

- Use `git status` frequently
- Take screenshots at each step
- Test in clean environment before submission
- Keep `mlruns/` for experiment history

---

## 🆘 Troubleshooting

**Import errors?**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Data not found?**
```bash
python src/data/download_data.py
```

**API won't start?**
```bash
# Check if model exists
ls models/best_model.joblib
# Train if missing
python src/models/train.py
```

---

**Ready to submit? Run final checks:**
```bash
pytest tests/ -v
./scripts/test_api.sh
kubectl get all
```

🎓 **Good luck with your submission!**
