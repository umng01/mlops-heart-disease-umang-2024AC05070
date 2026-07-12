#!/bin/bash

# Kubernetes Deployment Script
# Author: Umang Sharma (2024AC05070)

set -e

echo "=========================================="
echo "Deploying to Kubernetes"
echo "=========================================="
echo ""

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed"
    echo "Please install kubectl: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

echo "✓ kubectl found"

# Check cluster connection
echo ""
echo "Checking cluster connection..."
kubectl cluster-info

# Apply Kubernetes manifests
echo ""
echo "Applying Kubernetes manifests..."
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

echo ""
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/heart-disease-api

echo ""
echo "✅ Deployment complete!"
echo ""

# Show deployment status
echo "Deployment status:"
kubectl get deployments
echo ""
echo "Pods:"
kubectl get pods
echo ""
echo "Services:"
kubectl get services
echo ""

# Get service URL
echo "Getting service URL..."
SERVICE_TYPE=$(kubectl get service heart-disease-api -o jsonpath='{.spec.type}')

if [ "$SERVICE_TYPE" == "LoadBalancer" ]; then
    echo "Service type: LoadBalancer"
    echo "Waiting for external IP..."
    kubectl get service heart-disease-api -w &
    sleep 5
    EXTERNAL_IP=$(kubectl get service heart-disease-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ ! -z "$EXTERNAL_IP" ]; then
        echo ""
        echo "✅ API is available at: http://${EXTERNAL_IP}"
        echo ""
        echo "Test the API:"
        echo "  curl http://${EXTERNAL_IP}/health"
    else
        echo ""
        echo "⚠ External IP not yet assigned. Check with:"
        echo "  kubectl get service heart-disease-api"
    fi
elif [ "$SERVICE_TYPE" == "NodePort" ]; then
    NODE_PORT=$(kubectl get service heart-disease-api -o jsonpath='{.spec.ports[0].nodePort}')
    echo ""
    echo "✅ API is available at: http://localhost:${NODE_PORT}"
    echo ""
    echo "Test the API:"
    echo "  curl http://localhost:${NODE_PORT}/health"
else
    echo ""
    echo "Service type: ${SERVICE_TYPE}"
    echo "Check service with: kubectl get service heart-disease-api"
fi

echo ""
echo "To view logs:"
echo "  kubectl logs -f deployment/heart-disease-api"
echo ""
echo "To delete deployment:"
echo "  kubectl delete -f deployment/kubernetes/"
echo ""
