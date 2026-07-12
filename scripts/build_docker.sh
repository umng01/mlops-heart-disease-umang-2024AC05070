#!/bin/bash

# Docker Build Script
# Author: Umang Sharma (2024AC05070)

set -e

IMAGE_NAME="heart-disease-api"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "Building Docker Image"
echo "=========================================="
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""

# Build Docker image
echo "Building image..."
docker build -t ${FULL_IMAGE} .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "Image details:"
docker images | grep ${IMAGE_NAME}
echo ""
echo "To run the container:"
echo "  docker run -p 8000:8000 ${FULL_IMAGE}"
echo ""
echo "To test the API:"
echo "  curl http://localhost:8000/health"
echo ""
