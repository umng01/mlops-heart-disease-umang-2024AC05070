# Heart Disease Prediction - Architecture Documentation

This document provides comprehensive architecture diagrams for the MLOps Heart Disease Prediction project.

## Table of Contents

1. [High-Level System Architecture](#high-level-system-architecture)
2. [Data Pipeline Flow](#data-pipeline-flow)
3. [Training Pipeline with MLflow](#training-pipeline-with-mlflow)
4. [API Architecture](#api-architecture)
5. [Docker Containerization](#docker-containerization)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Monitoring Stack](#monitoring-stack)
8. [CI/CD Workflow](#cicd-workflow)
9. [Complete End-to-End Flow](#complete-end-to-end-flow)

---

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        UCI[UCI ML Repository]
        RawData[(Raw Data<br/>heart.csv)]
        ProcessedData[(Processed Data<br/>train/test)]
    end

    subgraph "ML Pipeline"
        Download[Data Download<br/>download_data.py]
        EDA[Exploratory Analysis<br/>Jupyter Notebooks]
        Preprocess[Preprocessing<br/>preprocess.py]
        Train[Model Training<br/>train.py]
        MLflow[MLflow Tracking<br/>Experiment Management]
    end

    subgraph "Model Storage"
        ModelRegistry[(Model Registry)]
        BestModel[Best Model<br/>model.pkl]
        Pipeline[Preprocessing Pipeline<br/>pipeline.pkl]
    end

    subgraph "Serving Layer"
        API[FastAPI Application<br/>app.py]
        Prediction[Prediction Service<br/>predict.py]
    end

    subgraph "Deployment"
        Docker[Docker Container]
        K8s[Kubernetes Cluster<br/>2+ Replicas]
        LoadBalancer[Load Balancer<br/>Service]
    end

    subgraph "Monitoring"
        Prometheus[Prometheus<br/>Metrics Collection]
        Grafana[Grafana<br/>Dashboards]
        Logs[Application Logs]
    end

    subgraph "CI/CD"
        GitHub[GitHub Repository]
        Actions[GitHub Actions<br/>CI/CD Pipeline]
        Tests[Automated Tests<br/>Pytest]
    end

    UCI --> Download
    Download --> RawData
    RawData --> EDA
    EDA --> Preprocess
    Preprocess --> ProcessedData
    ProcessedData --> Train
    Train --> MLflow
    MLflow --> ModelRegistry
    ModelRegistry --> BestModel
    Train --> Pipeline
    BestModel --> Prediction
    Pipeline --> Prediction
    Prediction --> API
    API --> Docker
    Docker --> K8s
    K8s --> LoadBalancer
    API --> Prometheus
    Prometheus --> Grafana
    API --> Logs
    GitHub --> Actions
    Actions --> Tests
    Tests --> Docker
    Docker -.-> K8s

    style UCI fill:#e1f5ff
    style MLflow fill:#fff3e0
    style API fill:#e8f5e9
    style K8s fill:#f3e5f5
    style Prometheus fill:#fce4ec
```

---

## Data Pipeline Flow

```mermaid
flowchart LR
    subgraph "Data Acquisition"
        A[UCI ML Repository<br/>Heart Disease Dataset] -->|Download| B[data/raw/heart.csv<br/>303 samples, 14 features]
    end

    subgraph "Data Validation"
        B --> C{Data Quality<br/>Check}
        C -->|Missing Values?| D[Imputation Strategy]
        C -->|Outliers?| E[Statistical Analysis]
        C -->|Valid| F[Data Profiling]
    end

    subgraph "Preprocessing"
        D --> G[Feature Engineering]
        E --> G
        F --> G
        G --> H[Standard Scaling<br/>StandardScaler]
        H --> I[Train/Test Split<br/>80/20]
    end

    subgraph "Storage"
        I --> J[(data/processed/train.csv)]
        I --> K[(data/processed/test.csv)]
        H --> L[(preprocessing_pipeline.pkl)]
    end

    subgraph "Features"
        M[Clinical Features<br/>• Age<br/>• Sex<br/>• Chest Pain Type<br/>• Blood Pressure<br/>• Cholesterol<br/>• Blood Sugar<br/>• ECG Results<br/>• Heart Rate<br/>• Exercise Angina<br/>• ST Depression<br/>• Slope<br/>• Vessels<br/>• Thalassemia]
    end

    M -.->|Input Schema| B

    style A fill:#e1f5ff
    style J fill:#c8e6c9
    style K fill:#c8e6c9
    style L fill:#fff9c4
```

---

## Training Pipeline with MLflow

```mermaid
flowchart TB
    subgraph "Data Input"
        A[(Processed Training Data)]
    end

    subgraph "Model Selection"
        A --> B[Logistic Regression]
        A --> C[Random Forest]
        A --> D[XGBoost]
    end

    subgraph "Hyperparameter Tuning"
        B --> E1[GridSearchCV<br/>• C: [0.001, 0.01, 0.1, 1, 10]<br/>• penalty: l2<br/>• solver: liblinear]
        C --> E2[GridSearchCV<br/>• n_estimators: [100, 200]<br/>• max_depth: [10, 20, None]<br/>• min_samples_split: [2, 5, 10]]
        D --> E3[GridSearchCV<br/>• learning_rate: [0.01, 0.1, 0.3]<br/>• max_depth: [3, 5, 7]<br/>• n_estimators: [100, 200]]
    end

    subgraph "Cross-Validation"
        E1 --> F1[5-Fold CV<br/>Stratified]
        E2 --> F2[5-Fold CV<br/>Stratified]
        E3 --> F3[5-Fold CV<br/>Stratified]
    end

    subgraph "MLflow Experiment Tracking"
        F1 --> G[MLflow Run 1]
        F2 --> H[MLflow Run 2]
        F3 --> I[MLflow Run 3]

        G --> J[Log Parameters]
        H --> J
        I --> J

        G --> K[Log Metrics<br/>• Accuracy<br/>• Precision<br/>• Recall<br/>• F1-Score<br/>• ROC-AUC]
        H --> K
        I --> K

        G --> L[Log Artifacts<br/>• Confusion Matrix<br/>• ROC Curve<br/>• Feature Importance<br/>• Training Plots]
        H --> L
        I --> L

        G --> M[Register Model]
        H --> M
        I --> M
    end

    subgraph "Model Selection"
        K --> N{Compare Metrics}
        N -->|Best ROC-AUC| O[Select Best Model<br/>XGBoost ~0.95 AUC]
    end

    subgraph "Model Persistence"
        O --> P[Serialize Model<br/>model.pkl]
        O --> Q[Save Metadata<br/>model_info.json]
        P --> R[(models/ directory)]
        Q --> R
    end

    subgraph "MLflow UI"
        J -.-> S[Experiment View<br/>localhost:5000]
        K -.-> S
        L -.-> S
        M -.-> T[(Model Registry)]
    end

    style O fill:#4caf50,color:#fff
    style K fill:#fff3e0
    style S fill:#e1f5ff
    style R fill:#f3e5f5
```

---

## API Architecture

```mermaid
flowchart TB
    subgraph "Client Layer"
        A[HTTP Client<br/>curl/Postman/Browser]
    end

    subgraph "FastAPI Application"
        B[FastAPI<br/>ASGI Server<br/>Uvicorn]
        
        subgraph "Endpoints"
            C[GET /health<br/>Health Check]
            D[POST /predict<br/>Prediction Endpoint]
            E[GET /docs<br/>Swagger UI]
            F[GET /redoc<br/>ReDoc]
        end

        subgraph "Request Processing"
            G[Pydantic Validation<br/>PredictionRequest Schema]
            H[Input Validation<br/>• Type Checking<br/>• Range Validation<br/>• Required Fields]
        end

        subgraph "Business Logic"
            I[Load Model<br/>@lru_cache]
            J[Load Preprocessing Pipeline<br/>@lru_cache]
            K[Preprocess Input<br/>StandardScaler]
            L[Model Inference<br/>predict_proba]
            M[Format Response<br/>PredictionResponse]
        end

        subgraph "Response Generation"
            N[JSON Response<br/>• prediction<br/>• probability<br/>• confidence<br/>• label]
        end
    end

    subgraph "Model Storage"
        O[(models/model.pkl)]
        P[(models/preprocessing_pipeline.pkl)]
    end

    subgraph "Monitoring"
        Q[Prometheus Metrics<br/>• Request Count<br/>• Response Time<br/>• Error Rate<br/>• Prediction Distribution]
        R[Application Logs<br/>• INFO<br/>• WARNING<br/>• ERROR]
    end

    A -->|HTTP Request| B
    B --> C
    B --> D
    B --> E
    B --> F
    D --> G
    G --> H
    H --> I
    H --> J
    I --> O
    J --> P
    O --> K
    P --> K
    K --> L
    L --> M
    M --> N
    N -->|HTTP Response| A
    D -.-> Q
    D -.-> R

    style B fill:#e8f5e9
    style G fill:#fff3e0
    style L fill:#e1f5ff
    style Q fill:#fce4ec
```

### API Request/Response Schema

```mermaid
classDiagram
    class PredictionRequest {
        +int age
        +int sex
        +int cp
        +int trestbps
        +int chol
        +int fbs
        +int restecg
        +int thalach
        +int exang
        +float oldpeak
        +int slope
        +int ca
        +int thal
        +validate_features()
    }

    class PredictionResponse {
        +int prediction
        +str prediction_label
        +float probability
        +str confidence
        +calculate_confidence()
    }

    class HealthResponse {
        +str status
        +bool model_loaded
        +str version
    }

    PredictionRequest --> PredictionResponse : transforms to
```

---

## Docker Containerization

```mermaid
flowchart TB
    subgraph "Build Context"
        A[Source Code<br/>src/]
        B[Requirements<br/>requirements.txt]
        C[Models<br/>models/]
        D[Dockerfile]
    end

    subgraph "Docker Build Process"
        E[Base Image<br/>python:3.9-slim]
        F[Install System Dependencies<br/>apt-get]
        G[Copy Requirements<br/>COPY requirements.txt]
        H[Install Python Dependencies<br/>pip install]
        I[Copy Application Code<br/>COPY src/ models/]
        J[Set Working Directory<br/>WORKDIR /app]
        K[Expose Port<br/>EXPOSE 8000]
        L[Define Entrypoint<br/>CMD uvicorn]
    end

    subgraph "Docker Image"
        M[heart-disease-api:latest<br/>Size: ~500MB<br/>Layers: Multi-stage]
    end

    subgraph "Docker Container Runtime"
        N[Container Instance<br/>Port: 8000<br/>Memory: 512MB<br/>CPU: 0.5 cores]
        
        O[Environment Variables<br/>• MODEL_PATH<br/>• LOG_LEVEL<br/>• PROMETHEUS_MULTIPROC_DIR<br/>• ENVIRONMENT]
        
        P[Volume Mounts<br/>• /app/models<br/>• /app/logs<br/>• /app/data]
        
        Q[Health Check<br/>curl -f http://localhost:8000/health<br/>Interval: 30s<br/>Timeout: 10s<br/>Retries: 3]
    end

    subgraph "Container Lifecycle"
        R[Start Container<br/>docker run]
        S[Health Check Passes]
        T[Ready to Serve]
        U[Graceful Shutdown<br/>SIGTERM]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F --> G --> H --> I --> J --> K --> L
    L --> M
    M --> N
    N --> O
    N --> P
    N --> Q
    R --> N
    N --> S
    S --> T
    T --> U

    style M fill:#1976d2,color:#fff
    style T fill:#4caf50,color:#fff
    style Q fill:#ff9800,color:#fff
```

### Docker Compose Stack

```mermaid
graph TB
    subgraph "Docker Compose Services"
        subgraph "API Service"
            A[heart-disease-api<br/>Port: 8000<br/>Build: ./Dockerfile]
        end

        subgraph "Monitoring Services"
            B[Prometheus<br/>Port: 9090<br/>Image: prom/prometheus]
            C[Grafana<br/>Port: 3000<br/>Image: grafana/grafana]
        end

        subgraph "Shared Resources"
            D[(prometheus-data<br/>Volume)]
            E[(grafana-data<br/>Volume)]
            F[mlops-network<br/>Bridge Network]
        end

        subgraph "Configuration"
            G[prometheus.yml<br/>Scrape Configs]
            H[alerts.yml<br/>Alert Rules]
            I[grafana/provisioning<br/>Datasources & Dashboards]
        end
    end

    A -->|depends_on| B
    C -->|depends_on| B
    A -->|network| F
    B -->|network| F
    C -->|network| F
    B -->|volume| D
    C -->|volume| E
    B -->|config| G
    B -->|config| H
    C -->|config| I
    A -->|scrape_target| B
    B -->|datasource| C

    style A fill:#4caf50,color:#fff
    style B fill:#e65100,color:#fff
    style C fill:#f57c00,color:#fff
```

---

## Kubernetes Deployment

```mermaid
flowchart TB
    subgraph "Kubernetes Cluster"
        subgraph "Control Plane"
            A[API Server]
            B[Scheduler]
            C[Controller Manager]
            D[etcd]
        end

        subgraph "Namespace: default"
            subgraph "Deployment: heart-disease-prediction"
                E[ReplicaSet<br/>Desired: 2]
                
                subgraph "Pod 1"
                    F1[Container<br/>heart-disease-api<br/>Port: 8000]
                    F2[Liveness Probe<br/>/health<br/>30s initial delay]
                    F3[Readiness Probe<br/>/health<br/>10s initial delay]
                end

                subgraph "Pod 2"
                    G1[Container<br/>heart-disease-api<br/>Port: 8000]
                    G2[Liveness Probe<br/>/health<br/>30s initial delay]
                    G3[Readiness Probe<br/>/health<br/>10s initial delay]
                end

                E --> F1
                E --> G1
                F1 --> F2
                F1 --> F3
                G1 --> G2
                G1 --> G3
            end

            subgraph "Service: heart-disease-predictor-service"
                H[Type: LoadBalancer<br/>Port: 80<br/>TargetPort: 8000]
                I[Selector: app=heart-disease-predictor]
            end

            subgraph "Resource Limits"
                J[Requests<br/>CPU: 250m<br/>Memory: 256Mi]
                K[Limits<br/>CPU: 500m<br/>Memory: 512Mi]
            end
        end

        subgraph "External Access"
            L[External Load Balancer<br/>Public IP]
        end

        subgraph "Auto-Scaling Optional"
            M[Horizontal Pod Autoscaler<br/>Min: 2<br/>Max: 10<br/>Target CPU: 80%]
        end
    end

    subgraph "Client Traffic"
        N[External Users<br/>HTTP Requests]
    end

    subgraph "Image Registry"
        O[Docker Hub / Registry<br/>heart-disease-prediction:latest]
    end

    A --> B
    B --> C
    C --> D
    H --> I
    I --> F1
    I --> G1
    H --> L
    L --> N
    O -.->|Pull Image| F1
    O -.->|Pull Image| G1
    M -.->|Scale| E
    J --> F1
    K --> F1
    J --> G1
    K --> G1

    style E fill:#e8f5e9
    style H fill:#e1f5ff
    style L fill:#fce4ec
    style M fill:#fff3e0
```

### Kubernetes Manifest Structure

```mermaid
graph LR
    subgraph "Kubernetes Resources"
        A[deployment.yaml<br/>• Deployment<br/>• Replicas: 2<br/>• Health Probes<br/>• Resource Limits]
        B[service.yaml<br/>• Service<br/>• Type: LoadBalancer<br/>• Port Mapping]
        C[Optional: hpa.yaml<br/>• HorizontalPodAutoscaler<br/>• CPU-based scaling]
        D[Optional: ingress.yaml<br/>• Ingress<br/>• Domain routing]
    end

    subgraph "Apply to Cluster"
        E[kubectl apply -f]
    end

    A --> E
    B --> E
    C -.-> E
    D -.-> E

    style A fill:#4caf50,color:#fff
    style B fill:#2196f3,color:#fff
```

---

## Monitoring Stack

```mermaid
flowchart TB
    subgraph "Application Layer"
        A[FastAPI Application<br/>Prometheus Instrumentation]
    end

    subgraph "Metrics Collection"
        B[Prometheus Client Library<br/>• Counter<br/>• Gauge<br/>• Histogram<br/>• Summary]
        
        C[Exposed Metrics<br/>• http_requests_total<br/>• http_request_duration_seconds<br/>• prediction_count<br/>• model_inference_time<br/>• error_rate]
    end

    subgraph "Prometheus Server"
        D[Scrape Configuration<br/>Interval: 15s]
        E[Time Series Database<br/>TSDB]
        F[Alert Manager<br/>Alert Rules]
        G[Query Engine<br/>PromQL]
    end

    subgraph "Visualization"
        H[Grafana<br/>Port: 3000]
        
        subgraph "Dashboards"
            I[API Performance<br/>• Request Rate<br/>• Response Time<br/>• Error Rate<br/>• Status Codes]
            J[Model Metrics<br/>• Prediction Count<br/>• Prediction Distribution<br/>• Inference Latency<br/>• Confidence Scores]
            K[System Resources<br/>• CPU Usage<br/>• Memory Usage<br/>• Disk I/O<br/>• Network Traffic]
        end
    end

    subgraph "Alerting"
        L[Alert Rules<br/>• High Error Rate<br/>• Slow Response Time<br/>• Service Down<br/>• High CPU/Memory]
        M[Notification Channels<br/>• Email<br/>• Slack<br/>• PagerDuty]
    end

    subgraph "Logging"
        N[Application Logs<br/>JSON Format<br/>Structured Logging]
        O[Log Aggregation<br/>Optional: ELK Stack]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    G --> H
    H --> I
    H --> J
    H --> K
    F --> L
    L --> M
    A --> N
    N -.-> O

    style A fill:#e8f5e9
    style E fill:#fce4ec
    style H fill:#fff3e0
    style L fill:#ff9800,color:#fff
```

### Prometheus Metrics Example

```mermaid
graph LR
    subgraph "Custom Metrics"
        A[predictions_total<br/>Type: Counter<br/>Labels: model, outcome]
        B[prediction_probability<br/>Type: Histogram<br/>Buckets: 0.1 to 1.0]
        C[model_load_time<br/>Type: Gauge<br/>Unit: seconds]
        D[request_duration<br/>Type: Histogram<br/>Buckets: 0.005 to 10s]
    end

    style A fill:#4caf50,color:#fff
    style B fill:#2196f3,color:#fff
    style C fill:#ff9800,color:#fff
    style D fill:#9c27b0,color:#fff
```

---

## CI/CD Workflow

```mermaid
flowchart TB
    subgraph "Trigger Events"
        A[Git Push<br/>main/develop branch]
        B[Pull Request<br/>Created/Updated]
    end

    subgraph "GitHub Actions Workflow"
        subgraph "Stage 1: Lint"
            C[Checkout Code]
            D[Setup Python 3.9]
            E[Install flake8]
            F[Run Linting<br/>• Syntax Check<br/>• Code Style<br/>• Complexity]
        end

        subgraph "Stage 2: Test"
            G[Install Dependencies<br/>pip install -r requirements.txt]
            H[Run Pytest<br/>• Unit Tests<br/>• Integration Tests<br/>• API Tests]
            I[Generate Coverage Report<br/>--cov=src]
            J[Upload to Codecov<br/>Optional]
            K[Archive Coverage HTML]
        end

        subgraph "Stage 3: Build"
            L[Setup Docker Buildx]
            M[Build Docker Image<br/>Multi-stage Build]
            N[Tag Image<br/>github.sha]
            O[Test Docker Image<br/>Import Test]
            P[Cache Layers<br/>GitHub Actions Cache]
        end

        subgraph "Stage 4: Push Main Branch Only"
            Q{Is Main Branch?}
            R[Login to Docker Hub]
            S[Extract Metadata<br/>Tags & Labels]
            T[Build & Push Image<br/>• latest<br/>• sha tag<br/>• branch tag]
            U[Output Image Digest]
        end

        subgraph "Optional: Deploy"
            V[Update K8s Manifests<br/>Optional CD]
            W[Deploy to Staging<br/>Optional]
            X[Deploy to Production<br/>Manual Approval]
        end
    end

    subgraph "Artifacts & Reports"
        Y[Coverage Report HTML]
        Z[Test Results XML]
        AA[Docker Image<br/>Docker Hub Registry]
    end

    A --> C
    B --> C
    C --> D --> E --> F
    F -->|Success| G
    G --> H --> I
    I --> J
    I --> K
    H -->|Success| L
    L --> M --> N --> O --> P
    O -->|Success| Q
    Q -->|Yes| R
    R --> S --> T --> U
    Q -->|No| AB[Skip Push]
    T --> AA
    U -.-> V
    V -.-> W
    W -.-> X
    K --> Y
    H --> Z

    style F fill:#ffd54f
    style H fill:#81c784
    style O fill:#64b5f6
    style T fill:#ba68c8
    style X fill:#ff8a65
```

### CI/CD Pipeline Stages

```mermaid
gantt
    title CI/CD Pipeline Timeline
    dateFormat  ss
    axisFormat %S
    section Lint
    Code Linting           :a1, 00, 15s
    section Test
    Unit Tests             :a2, 15, 30s
    Integration Tests      :a3, 45, 20s
    Coverage Report        :a4, 65, 10s
    section Build
    Docker Build           :a5, 75, 60s
    Image Test             :a6, 135, 15s
    section Push
    Registry Push (Main)   :a7, 150, 30s
    section Deploy
    Deploy to Staging      :crit, a8, 180, 20s
```

---

## Complete End-to-End Flow

```mermaid
flowchart TB
    subgraph "Phase 1: Data Acquisition & Preparation"
        A1[UCI Dataset Download<br/>heart.csv]
        A2[Exploratory Data Analysis<br/>Jupyter Notebooks]
        A3[Data Preprocessing<br/>Scaling, Split]
        A4[(Processed Data<br/>train.csv, test.csv)]
    end

    subgraph "Phase 2: Model Development"
        B1[Model Training<br/>3 Algorithms]
        B2[Hyperparameter Tuning<br/>GridSearchCV]
        B3[Cross-Validation<br/>5-Fold Stratified]
        B4[MLflow Experiment Tracking<br/>Parameters, Metrics, Artifacts]
        B5[Model Selection<br/>Best: XGBoost]
        B6[(Model Artifacts<br/>model.pkl, pipeline.pkl)]
    end

    subgraph "Phase 3: API Development"
        C1[FastAPI Application<br/>REST Endpoints]
        C2[Pydantic Schemas<br/>Request/Response Validation]
        C3[Model Integration<br/>Load & Predict]
        C4[Health Checks<br/>Readiness/Liveness]
    end

    subgraph "Phase 4: Testing"
        D1[Unit Tests<br/>pytest]
        D2[Integration Tests<br/>API Testing]
        D3[Coverage Analysis<br/>90%+ Coverage]
    end

    subgraph "Phase 5: Containerization"
        E1[Dockerfile Creation<br/>Multi-stage Build]
        E2[Docker Build<br/>Optimized Image]
        E3[Docker Compose<br/>Multi-service Stack]
        E4[Local Testing<br/>Port 8000]
    end

    subgraph "Phase 6: CI/CD"
        F1[GitHub Actions<br/>Workflow Definition]
        F2[Automated Testing<br/>On Push/PR]
        F3[Docker Image Build<br/>Automated]
        F4[Registry Push<br/>Docker Hub]
    end

    subgraph "Phase 7: Orchestration"
        G1[Kubernetes Manifests<br/>deployment.yaml, service.yaml]
        G2[Deploy to Cluster<br/>kubectl apply]
        G3[Service Exposure<br/>LoadBalancer]
        G4[Auto-scaling<br/>HPA]
    end

    subgraph "Phase 8: Monitoring"
        H1[Prometheus<br/>Metrics Collection]
        H2[Grafana Dashboards<br/>Visualization]
        H3[Alert Rules<br/>Notifications]
        H4[Log Aggregation<br/>Structured Logs]
    end

    subgraph "Phase 9: Production"
        I1[Live API<br/>Public Endpoint]
        I2[Prediction Service<br/>Real-time Inference]
        I3[Health Monitoring<br/>24/7 Uptime]
        I4[Performance Optimization<br/>Continuous Improvement]
    end

    A1 --> A2 --> A3 --> A4
    A4 --> B1
    B1 --> B2 --> B3 --> B4
    B4 --> B5 --> B6
    B6 --> C1
    C1 --> C2 --> C3 --> C4
    C3 --> D1 --> D2 --> D3
    D3 --> E1 --> E2 --> E3 --> E4
    E2 --> F1 --> F2 --> F3 --> F4
    F4 --> G1 --> G2 --> G3 --> G4
    G2 --> H1 --> H2
    H1 --> H3
    C1 --> H4
    G3 --> I1 --> I2
    H2 --> I3
    I3 --> I4

    style A1 fill:#e1f5ff
    style B5 fill:#fff3e0
    style C1 fill:#e8f5e9
    style E2 fill:#1976d2,color:#fff
    style G2 fill:#f3e5f5
    style H2 fill:#fce4ec
    style I2 fill:#4caf50,color:#fff
```

---

## Architecture Highlights

### Key Components

1. **Data Pipeline**
   - Automated download from UCI repository
   - Comprehensive EDA with Jupyter notebooks
   - Robust preprocessing with sklearn pipelines
   - Train/test split with stratification

2. **ML Pipeline**
   - Multiple algorithm evaluation (Logistic Regression, Random Forest, XGBoost)
   - Hyperparameter optimization with GridSearchCV
   - 5-fold cross-validation for robust evaluation
   - MLflow experiment tracking for reproducibility

3. **API Layer**
   - FastAPI for high-performance REST API
   - Pydantic for automatic validation
   - Async/await for scalability
   - Health endpoints for orchestration

4. **Containerization**
   - Multi-stage Docker builds for optimization
   - Docker Compose for local development
   - Volume mounts for model persistence
   - Health checks for reliability

5. **Orchestration**
   - Kubernetes deployment with 2+ replicas
   - LoadBalancer service for external access
   - Liveness and readiness probes
   - Resource limits for stability
   - Optional HPA for auto-scaling

6. **Monitoring**
   - Prometheus for metrics collection
   - Grafana for visualization
   - Custom application metrics
   - Alert manager for notifications

7. **CI/CD**
   - GitHub Actions workflow
   - Automated testing on every push
   - Docker image building and pushing
   - Optional deployment automation

### Technology Stack Summary

| Layer | Technologies |
|-------|-------------|
| **ML** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Tracking** | MLflow |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | Pytest, pytest-cov |
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Kubernetes, kubectl |
| **Monitoring** | Prometheus, Grafana |
| **CI/CD** | GitHub Actions |
| **Visualization** | Matplotlib, Seaborn, Plotly |

---

## Deployment Models

### 1. Local Development
```
Developer Machine → Python venv → FastAPI → localhost:8000
```

### 2. Docker Local
```
Developer Machine → Docker → Container → localhost:8000
```

### 3. Docker Compose Stack
```
Docker Compose → API Container + Prometheus + Grafana
```

### 4. Kubernetes (Minikube)
```
Minikube → K8s Deployment → Service → External Access
```

### 5. Cloud Kubernetes (Production)
```
Cloud Provider (AWS EKS/GCP GKE/Azure AKS) → K8s Cluster → 
LoadBalancer → Public Internet
```

---

## Performance Characteristics

### API Performance
- **Response Time:** <100ms (p50), <200ms (p99)
- **Throughput:** 100+ requests/second (single container)
- **Availability:** 99.9% with 2+ replicas

### Model Performance
- **Inference Time:** <50ms per prediction
- **ROC-AUC Score:** ~0.95
- **Accuracy:** ~90%
- **Memory Usage:** ~100MB

### Scaling Capabilities
- **Horizontal:** Auto-scale 2-10 pods based on CPU
- **Vertical:** Adjust resource limits per pod
- **Multi-region:** Deploy across availability zones

---

## Security Considerations

1. **API Security**
   - Input validation with Pydantic
   - Rate limiting (optional)
   - HTTPS in production (with Ingress)

2. **Container Security**
   - Non-root user in container
   - Minimal base image (python:slim)
   - Security scanning (optional)

3. **Kubernetes Security**
   - RBAC for access control
   - Network policies for isolation
   - Secrets management for credentials

4. **Monitoring Security**
   - Prometheus authentication (production)
   - Grafana admin password
   - Secure service-to-service communication

---

## Future Enhancements

1. **ML Pipeline**
   - Feature store integration
   - A/B testing framework
   - Model retraining automation
   - Drift detection

2. **API**
   - Authentication/Authorization (OAuth2, JWT)
   - Rate limiting
   - API versioning
   - GraphQL endpoint

3. **Monitoring**
   - Distributed tracing (Jaeger)
   - ELK stack for log aggregation
   - Anomaly detection
   - SLA monitoring

4. **Deployment**
   - Multi-region deployment
   - Blue-green deployment
   - Canary releases
   - Service mesh (Istio)

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Project:** Heart Disease Prediction MLOps  
**Author:** Umang Sharma (2024AC05070)  
**Course:** AIMLCZG523 - Machine Learning Operations  
**Institution:** BITS Pilani  
**Last Updated:** July 2026
