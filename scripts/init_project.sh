#!/bin/bash

# MLOps Heart Disease Project - Initialization Script
# Author: Umang Sharma (2024AC05070)

set -e  # Exit on error

echo "=========================================="
echo "MLOps Heart Disease Project Setup"
echo "Student: Umang Sharma (2024AC05070)"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python3 found${NC}"

# Create virtual environment
echo -e "\n${YELLOW}[1/7] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}[2/7] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}[3/7] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install requirements
echo -e "\n${YELLOW}[4/7] Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

# Create necessary directories
echo -e "\n${YELLOW}[5/7] Creating project directories...${NC}"
mkdir -p data/{raw,processed}
mkdir -p models
mkdir -p screenshots
mkdir -p logs
mkdir -p mlruns
echo -e "${GREEN}✓ Directories created${NC}"

# Download dataset
echo -e "\n${YELLOW}[6/7] Downloading dataset...${NC}"
if [ ! -f "data/raw/heart_disease.csv" ]; then
    python src/data/download_data.py
    echo -e "${GREEN}✓ Dataset downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Dataset already exists${NC}"
fi

# Initialize git (if not already initialized)
echo -e "\n${YELLOW}[7/7] Initializing Git repository...${NC}"
if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "Initial commit - MLOps Heart Disease Project by Umang Sharma (2024AC05070)"
    echo -e "${GREEN}✓ Git repository initialized${NC}"
else
    echo -e "${YELLOW}⚠ Git repository already initialized${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Project setup complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run EDA notebook: jupyter notebook notebooks/01_eda_preprocessing.ipynb"
echo "3. Train models: python src/models/train.py"
echo "4. Start API: uvicorn src.api.app:app --reload"
echo "5. View MLflow UI: mlflow ui"
echo ""
echo "For more details, see: README.md"
echo ""
