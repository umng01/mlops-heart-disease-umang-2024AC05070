"""
Data Download Script
Downloads Heart Disease UCI Dataset
Author: Umang Sharma (2024AC05070)
"""

import os
import pandas as pd
import requests
from pathlib import Path

def download_heart_disease_data():
    """Download Heart Disease UCI dataset"""

    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # UCI Heart Disease dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    try:
        print("Downloading Heart Disease dataset from UCI Repository...")
        response = requests.get(url)
        response.raise_for_status()

        # Save raw data
        raw_file = data_dir / "heart_disease_raw.csv"
        with open(raw_file, 'wb') as f:
            f.write(response.content)

        # Read and add column names
        df = pd.read_csv(raw_file, names=column_names, na_values='?')

        # Save with proper column names
        output_file = data_dir / "heart_disease.csv"
        df.to_csv(output_file, index=False)

        print(f"✓ Dataset downloaded successfully!")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Attempting alternative source...")

        # Alternative: Use a backup URL or create sample data
        # For robustness, you can add backup sources here
        raise

if __name__ == "__main__":
    download_heart_disease_data()
