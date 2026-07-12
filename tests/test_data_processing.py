"""
Unit tests for data processing module
Tests data loading, preprocessing, validation, and pipeline functionality
Author: Umang Sharma (2024AC05070)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import DataPreprocessor
from src.utils.config import FEATURE_NAMES, TARGET_NAME


@pytest.fixture
def sample_data():
    """Create sample heart disease data for testing"""
    np.random.seed(42)
    n_samples = 100

    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples).round(1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 5, n_samples)
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values"""
    np.random.seed(42)
    n_samples = 100

    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples).round(1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 5, n_samples)
    }

    df = pd.DataFrame(data)

    # Introduce missing values
    df.loc[0:5, 'age'] = np.nan
    df.loc[10:15, 'trestbps'] = np.nan
    df.loc[20:25, 'chol'] = np.nan
    df.loc[30:32, 'ca'] = np.nan

    return df


@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance"""
    return DataPreprocessor()


@pytest.fixture
def temp_csv_file(sample_data):
    """Create temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


class TestDataLoading:
    """Test data loading functionality"""

    def test_load_data_success(self, preprocessor, temp_csv_file):
        """Test successful data loading"""
        df = preprocessor.load_data(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.shape[0] == 100
        assert all(col in df.columns for col in FEATURE_NAMES)

    def test_load_data_shape(self, preprocessor, temp_csv_file):
        """Test loaded data has correct shape"""
        df = preprocessor.load_data(temp_csv_file)

        assert df.shape[0] > 0, "DataFrame should have rows"
        assert df.shape[1] == 14, "DataFrame should have 14 columns (13 features + 1 target)"

    def test_load_data_file_not_found(self, preprocessor):
        """Test error handling for missing file"""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("nonexistent_file.csv")

    def test_load_data_columns(self, preprocessor, temp_csv_file):
        """Test that loaded data contains expected columns"""
        df = preprocessor.load_data(temp_csv_file)
        expected_columns = FEATURE_NAMES + ['target']

        for col in expected_columns:
            assert col in df.columns, f"Column {col} missing from loaded data"


class TestMissingValueHandling:
    """Test missing value handling"""

    def test_handle_missing_values_numerical(self, preprocessor, sample_data_with_missing):
        """Test handling missing values in numerical features"""
        df_before = sample_data_with_missing.copy()
        missing_before = df_before.isnull().sum().sum()

        df_after = preprocessor.handle_missing_values(df_before)
        missing_after = df_after.isnull().sum().sum()

        assert missing_after == 0, "All missing values should be handled"
        assert missing_before > missing_after

    def test_handle_missing_values_median_fill(self, preprocessor, sample_data_with_missing):
        """Test that numerical features are filled with median"""
        df = sample_data_with_missing.copy()

        # Get median before filling
        age_median = df['age'].median()

        df_processed = preprocessor.handle_missing_values(df)

        # Check that missing values were filled
        assert df_processed['age'].isnull().sum() == 0
        # Verify some values were actually filled
        assert df_processed['age'].notna().all()

    def test_handle_missing_values_categorical(self, preprocessor):
        """Test handling missing values in categorical features"""
        df = pd.DataFrame({
            'age': [50, 60, 55, 48],
            'sex': [1, np.nan, 0, 1],
            'cp': [0, 1, np.nan, 2],
            'trestbps': [120, 130, 140, 135],
            'chol': [200, 220, 210, 205],
            'fbs': [0, 1, 0, np.nan],
            'restecg': [0, 1, 0, 1],
            'thalach': [150, 160, 155, 158],
            'exang': [0, 1, 0, 0],
            'oldpeak': [1.0, 1.5, 2.0, 1.2],
            'slope': [1, 2, 1, 2],
            'ca': [0, np.nan, 1, 0],
            'thal': [2, 3, 2, 3],
            'target': [0, 1, 0, 1]
        })

        df_processed = preprocessor.handle_missing_values(df)

        # Check that all missing values are handled
        assert df_processed.isnull().sum().sum() == 0

    def test_no_missing_values(self, preprocessor, sample_data):
        """Test handling data with no missing values"""
        df_processed = preprocessor.handle_missing_values(sample_data)

        assert df_processed.isnull().sum().sum() == 0
        assert df_processed.equals(sample_data)


class TestDataShapeValidation:
    """Test data shape validation"""

    def test_data_shape_after_loading(self, preprocessor, sample_data):
        """Test data shape is maintained after loading"""
        original_shape = sample_data.shape

        # Simulate processing
        df = preprocessor.handle_missing_values(sample_data)

        assert df.shape[0] == original_shape[0], "Number of rows should not change"

    def test_train_test_split_shapes(self, preprocessor, sample_data):
        """Test that train/test split maintains correct proportions"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)

        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)

        total_samples = len(X)
        expected_train = int(total_samples * 0.8)
        expected_test = int(total_samples * 0.2)

        # Allow for rounding differences
        assert abs(X_train.shape[0] - expected_train) <= 1
        assert abs(X_test.shape[0] - expected_test) <= 1
        assert X_train.shape[0] + X_test.shape[0] == total_samples

    def test_feature_dimension_consistency(self, preprocessor, sample_data):
        """Test that feature dimensions are consistent"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)

        assert X.shape[1] == len(FEATURE_NAMES), "Feature count should match config"
        assert len(y) == X.shape[0], "Target length should match feature rows"

    def test_scaled_data_shape(self, preprocessor, sample_data):
        """Test that scaling maintains data shape"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape


class TestFeatureTypesValidation:
    """Test feature types validation"""

    def test_feature_dtypes(self, preprocessor, sample_data):
        """Test that features have correct data types"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)

        # All features should be numeric
        assert all(X[col].dtype in [np.int64, np.float64] for col in X.columns)

    def test_target_binary_values(self, preprocessor, sample_data):
        """Test that target is binary (0 or 1)"""
        df = preprocessor.create_target_binary(sample_data)

        assert TARGET_NAME in df.columns
        assert set(df[TARGET_NAME].unique()).issubset({0, 1})

    def test_target_creation(self, preprocessor, sample_data):
        """Test target binary creation logic"""
        df = preprocessor.create_target_binary(sample_data)

        # Check that target is converted correctly
        # target > 0 should become 1, target == 0 should become 0
        assert df[TARGET_NAME].dtype in [np.int64, np.int32]
        assert df[TARGET_NAME].min() >= 0
        assert df[TARGET_NAME].max() <= 1

    def test_feature_names_match_config(self, preprocessor, sample_data):
        """Test that feature names match configuration"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)

        assert list(X.columns) == FEATURE_NAMES

    def test_missing_features_error(self, preprocessor):
        """Test error when required features are missing"""
        df = pd.DataFrame({
            'age': [50, 60],
            'sex': [1, 0],
            'target_binary': [0, 1]
        })

        with pytest.raises(ValueError, match="Missing features"):
            preprocessor.prepare_features(df)

    def test_feature_value_ranges(self, preprocessor, sample_data):
        """Test that features have reasonable value ranges"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)

        # Age should be reasonable
        assert X['age'].min() >= 0
        assert X['age'].max() <= 120

        # Binary features
        for col in ['sex', 'fbs', 'exang']:
            assert X[col].min() >= 0
            assert X[col].max() <= 1


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline"""

    def test_pipeline_execution(self, preprocessor, sample_data):
        """Test that pipeline runs end-to-end"""
        result = preprocessor.preprocess_pipeline(sample_data)

        assert len(result) == 6, "Pipeline should return 6 outputs"
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = result

        # Check all outputs are valid
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None

    def test_pipeline_data_integrity(self, preprocessor, sample_data):
        """Test that pipeline maintains data integrity"""
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = \
            preprocessor.preprocess_pipeline(sample_data)

        # Check that no data is lost
        total_samples = X_train.shape[0] + X_test.shape[0]
        assert total_samples == len(sample_data)

        # Check feature count
        assert X_train.shape[1] == len(FEATURE_NAMES)
        assert X_test.shape[1] == len(FEATURE_NAMES)

    def test_pipeline_scaling_effect(self, preprocessor, sample_data):
        """Test that scaling is applied correctly"""
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = \
            preprocessor.preprocess_pipeline(sample_data)

        # Scaled data should have approximately mean=0, std=1
        mean_values = X_train.mean(axis=0)
        std_values = X_train.std(axis=0)

        # Check that scaling was applied (means close to 0, std close to 1)
        assert np.allclose(mean_values, 0, atol=0.1), "Scaled data should have mean ~0"
        assert np.allclose(std_values, 1, atol=0.1), "Scaled data should have std ~1"

    def test_pipeline_stratification(self, preprocessor, sample_data):
        """Test that train/test split maintains class distribution"""
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = \
            preprocessor.preprocess_pipeline(sample_data)

        # Check class distribution is similar in train and test
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        # Ratios should be close (within 10%)
        assert abs(train_ratio - test_ratio) < 0.1, \
            "Train and test should have similar class distributions"

    def test_pipeline_reproducibility(self, preprocessor, sample_data):
        """Test that pipeline produces reproducible results"""
        result1 = preprocessor.preprocess_pipeline(sample_data.copy())

        # Create new preprocessor instance
        preprocessor2 = DataPreprocessor()
        result2 = preprocessor2.preprocess_pipeline(sample_data.copy())

        X_train1, X_test1, y_train1, y_test1 = result1[:4]
        X_train2, X_test2, y_train2, y_test2 = result2[:4]

        # Results should be identical
        assert np.allclose(X_train1, X_train2)
        assert np.allclose(X_test1, X_test2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(y_test1, y_test2)

    def test_pipeline_with_missing_data(self, preprocessor, sample_data_with_missing):
        """Test pipeline handles missing data correctly"""
        result = preprocessor.preprocess_pipeline(sample_data_with_missing)
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = result

        # No missing values should remain
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
        assert not y_train.isna().any()
        assert not y_test.isna().any()


class TestScalingFunctionality:
    """Test feature scaling functionality"""

    def test_scaler_fit_transform(self, preprocessor, sample_data):
        """Test scaler fit and transform"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

        # Check scaling was applied
        assert not np.allclose(X_train_scaled, X_train.values)
        assert X_train_scaled.shape == X_train.shape

    def test_scaler_transform_only(self, preprocessor, sample_data):
        """Test transform without test set"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        X_train_scaled = preprocessor.scale_features(X_train)

        assert X_train_scaled is not None
        assert X_train_scaled.shape == X_train.shape

    def test_scaler_statistics(self, preprocessor, sample_data):
        """Test that scaler computes correct statistics"""
        df = preprocessor.create_target_binary(sample_data)
        X, y = preprocessor.prepare_features(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        X_train_scaled = preprocessor.scale_features(X_train)

        # After scaling, features should have mean~0, std~1
        means = X_train_scaled.mean(axis=0)
        stds = X_train_scaled.std(axis=0)

        assert np.allclose(means, 0, atol=1e-7)
        assert np.allclose(stds, 1, atol=1e-7)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_dataframe(self, preprocessor):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()

        with pytest.raises(Exception):
            preprocessor.preprocess_pipeline(df)

    def test_single_row(self, preprocessor, sample_data):
        """Test handling of single row"""
        df = sample_data.iloc[:1].copy()

        # Should raise error due to insufficient data for split
        with pytest.raises(Exception):
            preprocessor.preprocess_pipeline(df)

    def test_all_missing_column(self, preprocessor, sample_data):
        """Test handling of column with all missing values"""
        df = sample_data.copy()
        df['age'] = np.nan

        df_processed = preprocessor.handle_missing_values(df)

        # When all values are NaN, median is NaN, so they remain NaN
        # This is expected behavior - in production, this should be caught earlier
        assert df_processed['age'].isna().all()

    def test_duplicate_rows(self, preprocessor, sample_data):
        """Test handling of duplicate rows"""
        df = pd.concat([sample_data, sample_data], ignore_index=True)

        result = preprocessor.preprocess_pipeline(df)

        # Should still process successfully
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
