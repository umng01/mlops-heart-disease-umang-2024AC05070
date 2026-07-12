# Complete Setup Guide

**Project:** Heart Disease Prediction MLOps Pipeline  
**Student:** Umang Sharma | **Roll No:** 2024AC05070

---

## Table of Contents

1. [Prerequisites Installation](#1-prerequisites-installation)
2. [Virtual Environment Setup](#2-virtual-environment-setup)
3. [Installing Dependencies](#3-installing-dependencies)
4. [Downloading Data](#4-downloading-data)
5. [Running Notebooks](#5-running-notebooks)
6. [Training Models](#6-training-models)
7. [Local API Testing](#7-local-api-testing)
8. [Docker Build and Run](#8-docker-build-and-run)
9. [Kubernetes Deployment](#9-kubernetes-deployment)
10. [Accessing Deployed Service](#10-accessing-deployed-service)
11. [Monitoring Setup](#11-monitoring-setup)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites Installation

### 1.1 Python 3.9+

**macOS:**
```bash
# Check current version
python3 --version

# If needed, install via Homebrew
brew install python@3.9

# Verify installation
python3 --version  # Should show 3.9.x or higher
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip
python3.9 --version
```

**Windows:**
1. Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
2. Run installer, check "Add Python to PATH"
3. Verify in Command Prompt:
   ```cmd
   python --version
   ```

### 1.2 Docker

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker

# Or download from: https://www.docker.com/products/docker-desktop

# Start Docker Desktop application
# Verify installation
docker --version
docker ps  # Should connect without errors
```

**Ubuntu/Debian:**
```bash
# Install Docker Engine
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker run hello-world
```

**Windows:**
1. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Enable WSL 2 backend if prompted
3. Verify in PowerShell:
   ```powershell
   docker --version
   ```

### 1.3 kubectl (Kubernetes CLI)

**macOS:**
```bash
# Using Homebrew
brew install kubectl

# Verify
kubectl version --client
```

**Ubuntu/Debian:**
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
```

**Windows:**
```powershell
# Using Chocolatey
choco install kubernetes-cli

# Or download from: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/
# Verify
kubectl version --client
```

### 1.4 Minikube (Local Kubernetes Cluster)

**macOS:**
```bash
brew install minikube

# Start Minikube
minikube start --driver=docker

# Verify
minikube status
kubectl cluster-info
```

**Ubuntu/Debian:**
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube start --driver=docker
minikube status
```

**Windows:**
```powershell
choco install minikube
minikube start --driver=docker
minikube status
```

### 1.5 Git (Version Control)

**macOS:**
```bash
brew install git
git --version
```

**Ubuntu/Debian:**
```bash
sudo apt install git
git --version
```

**Windows:**
Download from [git-scm.com](https://git-scm.com/download/win)

---

## 2. Virtual Environment Setup

### 2.1 Navigate to Project Directory

```bash
cd /Users/umang.sharma/mlops-heart-disease-project
```

### 2.2 Create Virtual Environment

**Option 1: Using venv (Recommended)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify activation (should show virtual environment name in prompt)
which python  # Should point to venv/bin/python
```

**Option 2: Using conda**
```bash
conda create -n heart-disease python=3.9
conda activate heart-disease
```

### 2.3 Upgrade pip and setuptools

```bash
pip install --upgrade pip setuptools wheel
```

---

## 3. Installing Dependencies

### 3.1 Install All Requirements

```bash
# Ensure virtual environment is activated
pip install -r requirements.txt
```

### 3.2 Verify Installation

```bash
# Check key packages
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import fastapi; print('fastapi:', fastapi.__version__)"
python -c "import mlflow; print('mlflow:', mlflow.__version__)"
```

### 3.3 Install Jupyter (Optional for Notebooks)

```bash
pip install jupyter jupyterlab
```

---

## 4. Downloading Data

### 4.1 Automatic Download

```bash
# Run data download script
python src/data/download_data.py
```

This will download the Heart Disease UCI dataset to `data/raw/heart.csv`

### 4.2 Manual Download (If Automatic Fails)

1. Visit: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
2. Download `heart.csv`
3. Place in `data/raw/heart.csv`:
   ```bash
   mkdir -p data/raw
   # Copy downloaded file
   cp ~/Downloads/heart.csv data/raw/
   ```

### 4.3 Verify Data

```bash
# Check if data exists
ls -lh data/raw/heart.csv

# Quick data inspection
python -c "import pandas as pd; df = pd.read_csv('data/raw/heart.csv'); print(df.shape); print(df.head())"
```

---

## 5. Running Notebooks

### 5.1 Start Jupyter Notebook

```bash
# Option 1: Classic Jupyter Notebook
jupyter notebook

# Option 2: JupyterLab (Modern Interface)
jupyter lab

# Browser should open automatically at http://localhost:8888
```

### 5.2 Run EDA Notebook

1. Navigate to `notebooks/` folder
2. Open `01_eda_preprocessing.ipynb`
3. Run cells sequentially:
   - Click "Run" or press `Shift+Enter` for each cell
   - Or select "Kernel > Restart & Run All"
4. Observe:
   - Dataset statistics
   - Visualizations (distributions, correlations)
   - Missing value analysis
   - Feature engineering

### 5.3 Run Model Training Notebook

1. Open `02_model_training.ipynb`
2. Run all cells
3. This notebook will:
   - Load preprocessed data
   - Train multiple models (Logistic Regression, Random Forest, XGBoost)
   - Compare performance
   - Save best model to `models/`
   - Log experiments to MLflow

---

## 6. Training Models

### 6.1 Train via Scripts (Alternative to Notebooks)

```bash
# Preprocess data
python src/data/preprocess.py

# Train models
python src/models/train.py
```

### 6.2 View MLflow Experiments

```bash
# Start MLflow UI
mlflow ui

# Open browser at: http://localhost:5000
```

**In MLflow UI:**
- View all experiment runs
- Compare metrics (accuracy, precision, recall, F1)
- View parameters and artifacts
- Download trained models

### 6.3 Check Trained Models

```bash
# List saved models
ls -lh models/

# Should contain:
# - best_model.joblib (Best performing model)
# - scaler.pkl (Data scaler)
# - model_metadata.json (Model info)
```

---

## 7. Local API Testing

### 7.1 Start FastAPI Server

```bash
# Ensure virtual environment is activated and model is trained
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 7.2 Access API Documentation

Open browser at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 7.3 Test Health Endpoint

```bash
# In another terminal
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 7.4 Test Prediction Endpoint

**Using curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Using provided test script:**
```bash
./scripts/test_api.sh http://localhost:8000
```

**Expected Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "high",
  "model_version": "1.0.0"
}
```

### 7.5 Test Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1},
      {"age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 187, "exang": 0, "oldpeak": 3.5, "slope": 0, "ca": 0, "thal": 2}
    ]
  }'
```

---

## 8. Docker Build and Run

### 8.1 Build Docker Image

**Using build script (Recommended):**
```bash
./scripts/build_docker.sh
```

**Manual build:**
```bash
docker build -t heart-disease-prediction:latest .
```

### 8.2 Verify Image

```bash
docker images | grep heart-disease
```

**Expected Output:**
```
heart-disease-prediction  latest  abc123def456  2 minutes ago  500MB
```

### 8.3 Run Docker Container

**Option 1: Simple run**
```bash
docker run -d \
  --name heart-disease-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  heart-disease-prediction:latest
```

**Option 2: Using docker-compose (Includes monitoring)**
```bash
docker-compose up -d
```

### 8.4 Check Container Status

```bash
docker ps

# View logs
docker logs heart-disease-api

# Follow logs
docker logs -f heart-disease-api
```

### 8.5 Test Dockerized API

```bash
# Health check
curl http://localhost:8000/health

# Prediction test
./scripts/test_api.sh http://localhost:8000
```

### 8.6 Stop and Remove Container

```bash
# Stop container
docker stop heart-disease-api

# Remove container
docker rm heart-disease-api

# Or with docker-compose
docker-compose down
```

---

## 9. Kubernetes Deployment

### 9.1 Ensure Minikube is Running

```bash
minikube status

# If not running
minikube start --driver=docker
```

### 9.2 Load Docker Image into Minikube

```bash
# Build image for Minikube
eval $(minikube docker-env)
docker build -t heart-disease-prediction:latest .

# Verify image in Minikube
minikube ssh docker images | grep heart-disease
```

### 9.3 Deploy to Kubernetes

**Using deployment script:**
```bash
./scripts/deploy_k8s.sh
```

**Manual deployment:**
```bash
# Apply deployment
kubectl apply -f deployment/kubernetes/deployment.yaml

# Apply service
kubectl apply -f deployment/kubernetes/service.yaml
```

### 9.4 Verify Deployment

```bash
# Check deployment status
kubectl get deployments

# Check pods
kubectl get pods

# Check services
kubectl get services

# View detailed pod info
kubectl describe pod <pod-name>
```

**Expected Output:**
```
NAME                          READY   STATUS    RESTARTS   AGE
heart-disease-prediction-xxx  1/1     Running   0          2m
```

### 9.5 Check Logs

```bash
# Get pod name
kubectl get pods

# View logs
kubectl logs <pod-name>

# Follow logs
kubectl logs -f <pod-name>
```

---

## 10. Accessing Deployed Service

### 10.1 Get Service URL (Minikube)

```bash
minikube service heart-disease-predictor-service --url
```

**Expected Output:**
```
http://192.168.49.2:30080
```

### 10.2 Port Forward (Alternative)

```bash
kubectl port-forward service/heart-disease-predictor-service 8000:80
```

Now access at: http://localhost:8000

### 10.3 Test Kubernetes Deployment

```bash
# Get service URL
SERVICE_URL=$(minikube service heart-disease-predictor-service --url)

# Health check
curl $SERVICE_URL/health

# Prediction test
./scripts/test_api.sh $SERVICE_URL
```

### 10.4 Access API Documentation

```bash
# Get URL
SERVICE_URL=$(minikube service heart-disease-predictor-service --url)
echo "Swagger UI: $SERVICE_URL/docs"
echo "ReDoc: $SERVICE_URL/redoc"
```

Open in browser.

---

## 11. Monitoring Setup

### 11.1 Start Monitoring Stack with Docker Compose

```bash
docker-compose up -d
```

This starts:
- **API:** Port 8000
- **Prometheus:** Port 9090
- **Grafana:** Port 3000

### 11.2 Access Prometheus

1. Open browser at: http://localhost:9090
2. Navigate to **Status > Targets** to verify API is being scraped
3. Run queries:
   - `up` - Check service status
   - `http_requests_total` - Total requests
   - `prediction_duration_seconds` - Prediction latency

### 11.3 Access Grafana

1. Open browser at: http://localhost:3000
2. Login:
   - **Username:** admin
   - **Password:** admin
3. Change password when prompted
4. Navigate to **Dashboards**
5. Open "Heart Disease Prediction API" dashboard

**Dashboard includes:**
- Request rate
- Error rate
- Response time (p50, p95, p99)
- Prediction distribution
- Model performance metrics

### 11.4 Custom Metrics

View application metrics at: http://localhost:8000/metrics

**Key metrics:**
- `predictions_total` - Total predictions made
- `prediction_duration_seconds` - Prediction latency histogram
- `model_predictions_by_outcome` - Predictions by class
- `api_requests_total` - HTTP request counter

### 11.5 Stop Monitoring

```bash
docker-compose down

# Remove volumes (clears data)
docker-compose down -v
```

---

## 12. Troubleshooting

### 12.1 Python/Environment Issues

**Problem:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

---

**Problem:** `Python version mismatch`

**Solution:**
```bash
# Check Python version
python --version

# Recreate virtual environment with correct version
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 12.2 Data Issues

**Problem:** `FileNotFoundError: data/raw/heart.csv`

**Solution:**
```bash
# Create directory
mkdir -p data/raw

# Download data
python src/data/download_data.py

# Or manually place heart.csv in data/raw/
```

---

**Problem:** `Corrupted or incomplete data`

**Solution:**
```bash
# Remove existing data
rm -rf data/raw/heart.csv

# Re-download
python src/data/download_data.py
```

---

### 12.3 Model Training Issues

**Problem:** `Model file not found`

**Solution:**
```bash
# Train model
python src/models/train.py

# Or run notebook
jupyter notebook notebooks/02_model_training.ipynb
```

---

**Problem:** `Memory error during training`

**Solution:**
```bash
# Reduce batch size or use smaller model
# Edit src/models/train.py
# Or train on smaller data subset
```

---

### 12.4 API Issues

**Problem:** `Connection refused on localhost:8000`

**Solution:**
```bash
# Check if API is running
ps aux | grep uvicorn

# If not, start it
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Check if port is in use
lsof -i :8000
```

---

**Problem:** `Model not loaded error`

**Solution:**
```bash
# Ensure model exists
ls models/best_model.joblib

# If missing, train model
python src/models/train.py

# Check MODEL_PATH environment variable
echo $MODEL_PATH
```

---

**Problem:** `422 Unprocessable Entity error`

**Solution:**
- Verify request JSON matches expected schema
- Check API docs at http://localhost:8000/docs
- Ensure all required fields are present
- Verify data types (integers vs floats)

---

### 12.5 Docker Issues

**Problem:** `Cannot connect to Docker daemon`

**Solution:**
```bash
# macOS: Start Docker Desktop application
open -a Docker

# Linux: Start Docker service
sudo systemctl start docker

# Verify
docker ps
```

---

**Problem:** `Docker build fails`

**Solution:**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t heart-disease-prediction:latest .
```

---

**Problem:** `Container exits immediately`

**Solution:**
```bash
# Check logs
docker logs heart-disease-api

# Run interactively to debug
docker run -it heart-disease-prediction:latest /bin/bash
```

---

### 12.6 Kubernetes Issues

**Problem:** `Minikube not starting`

**Solution:**
```bash
# Delete and recreate cluster
minikube delete
minikube start --driver=docker

# Check status
minikube status
```

---

**Problem:** `ImagePullBackOff error`

**Solution:**
```bash
# Ensure image is in Minikube's Docker
eval $(minikube docker-env)
docker build -t heart-disease-prediction:latest .

# Verify image
minikube ssh docker images | grep heart-disease

# Redeploy
kubectl delete -f deployment/kubernetes/
kubectl apply -f deployment/kubernetes/
```

---

**Problem:** `Pods not running`

**Solution:**
```bash
# Get pod name
kubectl get pods

# Describe pod for errors
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Common fix: Delete and recreate
kubectl delete pod <pod-name>
```

---

**Problem:** `Service not accessible`

**Solution:**
```bash
# Check service
kubectl get services

# Verify endpoints
kubectl get endpoints

# Use port-forward instead
kubectl port-forward service/heart-disease-predictor-service 8000:80
```

---

### 12.7 Monitoring Issues

**Problem:** `Prometheus not scraping metrics`

**Solution:**
```bash
# Check Prometheus targets
# Open http://localhost:9090/targets

# Verify API metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus config
cat monitoring/prometheus.yml
```

---

**Problem:** `Grafana dashboard not showing data`

**Solution:**
1. Verify Prometheus datasource in Grafana (Configuration > Data Sources)
2. Check time range in dashboard (top-right)
3. Ensure API has received requests (generate some traffic)
4. Verify metrics are being collected:
   ```bash
   curl http://localhost:8000/metrics
   ```

---

### 12.8 Permission Issues

**Problem:** `Permission denied when running scripts`

**Solution:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Or run with bash explicitly
bash scripts/build_docker.sh
```

---

### 12.9 Port Already in Use

**Problem:** `Port 8000 already in use`

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn src.api.app:app --port 8001
```

---

### 12.10 MLflow Issues

**Problem:** `MLflow UI not starting`

**Solution:**
```bash
# Ensure mlflow is installed
pip install mlflow

# Check if port 5000 is available
lsof -i :5000

# Use different port
mlflow ui --port 5001
```

---

### 12.11 General Debugging Steps

1. **Check Python environment:**
   ```bash
   which python
   pip list
   ```

2. **Verify project structure:**
   ```bash
   ls -la
   tree -L 2  # If tree is installed
   ```

3. **Check logs:**
   - Application logs: `logs/app.log`
   - Docker logs: `docker logs <container-name>`
   - Kubernetes logs: `kubectl logs <pod-name>`

4. **Test components individually:**
   ```bash
   # Test data loading
   python -c "from src.data.download_data import load_data; load_data()"
   
   # Test model loading
   python -c "from src.models.predict import load_model; load_model()"
   
   # Test API
   pytest tests/test_api.py -v
   ```

5. **Clean and restart:**
   ```bash
   # Clean Python cache
   find . -type d -name __pycache__ -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   
   # Restart virtual environment
   deactivate
   source venv/bin/activate
   ```

---

## Getting Help

If issues persist:

1. **Check project README:** `/Users/umang.sharma/mlops-heart-disease-project/README.md`
2. **Review logs carefully** for error messages
3. **Check documentation:**
   - FastAPI: https://fastapi.tiangolo.com/
   - Docker: https://docs.docker.com/
   - Kubernetes: https://kubernetes.io/docs/
4. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

---

## Quick Reference Commands

```bash
# Environment
source venv/bin/activate

# Data
python src/data/download_data.py

# Training
python src/models/train.py

# API (Local)
uvicorn src.api.app:app --reload

# Docker
docker build -t heart-disease-prediction:latest .
docker run -p 8000:8000 heart-disease-prediction:latest
docker-compose up -d

# Kubernetes
minikube start
kubectl apply -f deployment/kubernetes/
kubectl get pods
minikube service heart-disease-predictor-service --url

# Monitoring
mlflow ui  # Port 5000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000

# Testing
pytest tests/ -v
./scripts/test_api.sh http://localhost:8000
```

---

## Next Steps

After successful setup:

1. Explore API documentation: http://localhost:8000/docs
2. Run all unit tests: `pytest tests/ -v`
3. Review MLflow experiments: `mlflow ui`
4. Set up monitoring with docker-compose
5. Practice making predictions via API
6. Experiment with different models in notebooks

---

**Setup complete! You're ready to start developing and deploying ML models.**
