"""
Data Preprocessing Module
Author: Umang Sharma (2024AC05070)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    PROCESSED_DATA_DIR, RAW_DATA_DIR,
    FEATURE_NAMES, TARGET_NAME, TEST_SIZE, RANDOM_STATE
)


class DataPreprocessor:
    """Handles all data preprocessing operations"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = FEATURE_NAMES

    def load_data(self, filepath):
        """Load data from CSV file"""
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded: {df.shape}")
        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\n=== Handling Missing Values ===")

        # Numerical features
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if col in numerical_features:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"✓ {col}: Filled {df[col].isnull().sum()} values with median ({median_val:.2f})")
                else:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                    print(f"✓ {col}: Filled with mode ({mode_val})")

        print(f"✓ Total missing values: {df.isnull().sum().sum()}")
        return df

    def create_target_binary(self, df):
        """Convert target to binary (0: no disease, 1: disease)"""
        if 'target' in df.columns and TARGET_NAME not in df.columns:
            df[TARGET_NAME] = (df['target'] > 0).astype(int)
            print(f"✓ Binary target created: {df[TARGET_NAME].value_counts().to_dict()}")
        return df

    def prepare_features(self, df):
        """Prepare features and target for modeling"""
        # Ensure all required features exist
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Extract features and target
        X = df[self.feature_names].copy()
        y = df[TARGET_NAME].copy()

        print(f"\n✓ Features prepared: {X.shape}")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def split_data(self, X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\n✓ Data split:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Testing: {X_test.shape[0]} samples")
        print(f"  Train target: {y_train.value_counts().to_dict()}")
        print(f"  Test target: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def preprocess_pipeline(self, df):
        """Complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Create binary target
        df = self.create_target_binary(df)

        # Prepare features
        X, y = self.prepare_features(df)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test


def main():
    """Main preprocessing function"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load raw data
    raw_file = RAW_DATA_DIR / "heart_disease.csv"
    if not raw_file.exists():
        print("Error: Raw data file not found!")
        print("Please run: python src/data/download_data.py")
        return

    df = preprocessor.load_data(raw_file)

    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = preprocessor.preprocess_pipeline(df)

    # Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save train/test splits
    train_df = X_train_orig.copy()
    train_df['target'] = y_train.values
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)

    test_df = X_test_orig.copy()
    test_df['target'] = y_test.values
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print(f"\n✓ Processed data saved to: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
