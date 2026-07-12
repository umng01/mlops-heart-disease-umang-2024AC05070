#!/bin/bash
# Create complete project structure

# Main directories
mkdir -p data/{raw,processed}
mkdir -p notebooks
mkdir -p src/{data,models,api,utils}
mkdir -p tests
mkdir -p mlruns
mkdir -p models
mkdir -p deployment/{kubernetes,docker}
mkdir -p .github/workflows
mkdir -p screenshots
mkdir -p monitoring

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/api/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

echo "Project structure created successfully!"
