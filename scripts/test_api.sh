#!/bin/bash

# API Testing Script
# Author: Umang Sharma (2024AC05070)

API_URL="${1:-http://localhost:8000}"

echo "=========================================="
echo "Testing Heart Disease Prediction API"
echo "=========================================="
echo ""
echo "API URL: ${API_URL}"
echo ""

# Test health endpoint
echo "[1/3] Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s "${API_URL}/health")
echo "Response: ${HEALTH_RESPONSE}"

if echo "${HEALTH_RESPONSE}" | grep -q "healthy"; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

echo ""

# Test prediction endpoint with sample data
echo "[2/3] Testing /predict endpoint with sample data..."

# Sample patient data (high risk)
SAMPLE_DATA='{
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

echo "Input data:"
echo "${SAMPLE_DATA}" | python3 -m json.tool

echo ""
echo "Sending prediction request..."
PRED_RESPONSE=$(curl -s -X POST "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${SAMPLE_DATA}")

echo "Response:"
echo "${PRED_RESPONSE}" | python3 -m json.tool

if echo "${PRED_RESPONSE}" | grep -q "prediction"; then
    echo "✅ Prediction successful"
else
    echo "❌ Prediction failed"
    exit 1
fi

echo ""

# Test with another sample (low risk)
echo "[3/3] Testing with low risk patient..."

LOW_RISK_DATA='{
  "age": 45,
  "sex": 0,
  "cp": 0,
  "trestbps": 120,
  "chol": 180,
  "fbs": 0,
  "restecg": 0,
  "thalach": 170,
  "exang": 0,
  "oldpeak": 0.0,
  "slope": 2,
  "ca": 0,
  "thal": 2
}'

PRED_RESPONSE2=$(curl -s -X POST "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "${LOW_RISK_DATA}")

echo "Response:"
echo "${PRED_RESPONSE2}" | python3 -m json.tool

if echo "${PRED_RESPONSE2}" | grep -q "prediction"; then
    echo "✅ Prediction successful"
else
    echo "❌ Prediction failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ All API tests passed!"
echo "=========================================="
echo ""
