"""
Unit Tests for Model Loading and Prediction Functionality

This module tests the core model functionality including:
- Model loading from disk
- Prediction output shape
- Prediction probability ranges
- Input validation
- Preprocessing consistency

Author: Umang Sharma (2024AC05070)
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predict import HeartDiseasePredictor
from src.utils.config import FEATURE_NAMES, MODELS_DIR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def valid_single_sample():
    """Fixture providing a valid single sample"""
    return {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }


@pytest.fixture
def valid_batch_samples():
    """Fixture providing multiple valid samples"""
    return pd.DataFrame([
        {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        },
        {
            'age': 37, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 250,
            'fbs': 0, 'restecg': 1, 'thalach': 187, 'exang': 0,
            'oldpeak': 3.5, 'slope': 0, 'ca': 0, 'thal': 2
        },
        {
            'age': 41, 'sex': 0, 'cp': 1, 'trestbps': 130, 'chol': 204,
            'fbs': 0, 'restecg': 0, 'thalach': 172, 'exang': 0,
            'oldpeak': 1.4, 'slope': 2, 'ca': 0, 'thal': 2
        },
        {
            'age': 56, 'sex': 1, 'cp': 1, 'trestbps': 120, 'chol': 236,
            'fbs': 0, 'restecg': 1, 'thalach': 178, 'exang': 0,
            'oldpeak': 0.8, 'slope': 2, 'ca': 0, 'thal': 2
        },
        {
            'age': 57, 'sex': 0, 'cp': 0, 'trestbps': 120, 'chol': 354,
            'fbs': 0, 'restecg': 1, 'thalach': 163, 'exang': 1,
            'oldpeak': 0.6, 'slope': 2, 'ca': 0, 'thal': 2
        }
    ])


@pytest.fixture
def mock_pipeline():
    """Fixture providing a mock trained pipeline"""
    # Create a simple mock pipeline
    scaler = StandardScaler()
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)

    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', classifier)
    ])

    # Create dummy training data
    np.random.seed(42)
    X_train = np.random.randn(100, 13)
    y_train = np.random.randint(0, 2, 100)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    return pipeline


@pytest.fixture
def temp_model_file(tmp_path, mock_pipeline):
    """Fixture providing a temporary model file"""
    model_path = tmp_path / "test_model_pipeline.pkl"
    joblib.dump(mock_pipeline, model_path)
    return model_path


class TestModelLoading:
    """Test cases for model loading functionality"""

    def test_load_model_success(self, temp_model_file):
        """Test successful model loading from file"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        assert predictor.pipeline is not None
        assert isinstance(predictor.pipeline, Pipeline)
        assert 'scaler' in predictor.pipeline.named_steps
        assert 'classifier' in predictor.pipeline.named_steps

    def test_load_model_file_not_found(self):
        """Test model loading fails with non-existent file"""
        non_existent_path = "/path/to/nonexistent/model.pkl"

        with pytest.raises(FileNotFoundError) as exc_info:
            HeartDiseasePredictor(model_path=non_existent_path)

        assert "Model file not found" in str(exc_info.value)

    def test_load_model_auto_detect(self):
        """Test automatic model detection when no path provided"""
        try:
            predictor = HeartDiseasePredictor()
            assert predictor.pipeline is not None
        except FileNotFoundError:
            # This is expected if no model has been trained
            pytest.skip("No trained model available for auto-detection")

    def test_model_info_extraction(self, temp_model_file):
        """Test extraction of model information after loading"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        info = predictor.get_model_info()

        assert isinstance(info, dict)
        assert 'model_path' in info
        assert 'model_type' in info
        assert 'has_scaler' in info
        assert 'n_features' in info
        assert info['n_features'] == 13
        assert info['has_scaler'] is True

    def test_pipeline_components(self, temp_model_file):
        """Test that loaded pipeline has expected components"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        assert hasattr(predictor.pipeline, 'named_steps')
        assert 'scaler' in predictor.pipeline.named_steps
        assert 'classifier' in predictor.pipeline.named_steps

        # Verify scaler is StandardScaler
        assert isinstance(predictor.pipeline.named_steps['scaler'], StandardScaler)

    def test_model_path_stored_correctly(self, temp_model_file):
        """Test that model path is stored correctly"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        assert predictor.model_path == Path(temp_model_file)
        assert str(temp_model_file) in predictor.model_info['model_path']

    def test_feature_names_loaded(self, temp_model_file):
        """Test that feature names are loaded correctly"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        assert predictor.feature_names == FEATURE_NAMES
        assert len(predictor.feature_names) == 13


class TestPredictionShape:
    """Test cases for prediction output shapes"""

    def test_single_prediction_shape(self, temp_model_file, valid_single_sample):
        """Test prediction shape for single sample"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        predictions = predictor.predict(valid_single_sample, return_proba=False, validate=False)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (1,)
        assert predictions.ndim == 1

    def test_single_prediction_with_proba_shape(self, temp_model_file, valid_single_sample):
        """Test prediction shape with probabilities for single sample"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        predictions, probabilities = predictor.predict(
            valid_single_sample,
            return_proba=True,
            validate=False
        )

        assert isinstance(predictions, np.ndarray)
        assert isinstance(probabilities, np.ndarray)
        assert predictions.shape == (1,)
        assert probabilities.shape == (1, 2)
        assert probabilities.ndim == 2

    def test_batch_prediction_shape(self, temp_model_file, valid_batch_samples):
        """Test prediction shape for batch of samples"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        predictions = predictor.predict(
            valid_batch_samples,
            return_proba=False,
            validate=False
        )

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(valid_batch_samples),)
        assert len(predictions) == 5

    def test_batch_prediction_with_proba_shape(self, temp_model_file, valid_batch_samples):
        """Test prediction shape with probabilities for batch"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        predictions, probabilities = predictor.predict(
            valid_batch_samples,
            return_proba=True,
            validate=False
        )

        n_samples = len(valid_batch_samples)
        assert predictions.shape == (n_samples,)
        assert probabilities.shape == (n_samples, 2)

    def test_variable_batch_sizes(self, temp_model_file):
        """Test predictions work with different batch sizes"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        batch_sizes = [1, 3, 5, 10, 20]

        for size in batch_sizes:
            # Create batch of given size
            batch = pd.DataFrame({
                feature: np.random.randint(
                    predictor.FEATURE_RANGES.get(feature, (0, 100))[0],
                    predictor.FEATURE_RANGES.get(feature, (0, 100))[1] + 1,
                    size
                ) if feature != 'oldpeak' else np.random.uniform(0, 6, size)
                for feature in FEATURE_NAMES
            })

            predictions, probabilities = predictor.predict(
                batch,
                return_proba=True,
                validate=False
            )

            assert predictions.shape == (size,)
            assert probabilities.shape == (size, 2)

    def test_predict_single_method_output_format(self, temp_model_file, valid_single_sample):
        """Test predict_single method returns correct format"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        result = predictor.predict_single(valid_single_sample, validate=False)

        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'prediction_label' in result
        assert 'probability_positive' in result
        assert 'probability_negative' in result
        assert 'risk_level' in result
        assert 'confidence' in result
        assert 'input_features' in result

    def test_predict_batch_method_output_format(self, temp_model_file, valid_batch_samples):
        """Test predict_batch method returns correct format"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        result = predictor.predict_batch(valid_batch_samples, validate=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(valid_batch_samples)
        assert 'prediction' in result.columns
        assert 'prediction_label' in result.columns
        assert 'probability_positive' in result.columns
        assert 'probability_negative' in result.columns
        assert 'confidence' in result.columns
        assert 'risk_level' in result.columns


class TestPredictionRange:
    """Test cases for prediction probability ranges"""

    def test_predictions_are_binary(self, temp_model_file, valid_batch_samples):
        """Test that predictions are binary (0 or 1)"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        predictions = predictor.predict(valid_batch_samples, return_proba=False, validate=False)

        assert all(pred in [0, 1] for pred in predictions)
        assert predictions.dtype in [np.int32, np.int64, np.int_, int]

    def test_probabilities_in_valid_range(self, temp_model_file, valid_batch_samples):
        """Test that all probabilities are between 0 and 1"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        _, probabilities = predictor.predict(
            valid_batch_samples,
            return_proba=True,
            validate=False
        )

        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)

    def test_probabilities_sum_to_one(self, temp_model_file, valid_batch_samples):
        """Test that probabilities for each sample sum to 1"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        _, probabilities = predictor.predict(
            valid_batch_samples,
            return_proba=True,
            validate=False
        )

        row_sums = probabilities.sum(axis=1)
        assert np.allclose(row_sums, 1.0, rtol=1e-5)

    def test_negative_class_probability(self, temp_model_file, valid_single_sample):
        """Test that negative class probability is in valid range"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        result = predictor.predict_single(valid_single_sample, validate=False)

        prob_negative = result['probability_negative']
        assert 0.0 <= prob_negative <= 1.0
        assert isinstance(prob_negative, (float, np.floating))

    def test_positive_class_probability(self, temp_model_file, valid_single_sample):
        """Test that positive class probability is in valid range"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        result = predictor.predict_single(valid_single_sample, validate=False)

        prob_positive = result['probability_positive']
        assert 0.0 <= prob_positive <= 1.0
        assert isinstance(prob_positive, (float, np.floating))

    def test_confidence_score_range(self, temp_model_file, valid_batch_samples):
        """Test that confidence scores are in valid range"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        result = predictor.predict_batch(valid_batch_samples, validate=False)

        assert all(0.0 <= conf <= 1.0 for conf in result['confidence'])

    def test_confidence_is_max_probability(self, temp_model_file, valid_single_sample):
        """Test that confidence equals max probability"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        result = predictor.predict_single(valid_single_sample, validate=False)

        expected_confidence = max(
            result['probability_positive'],
            result['probability_negative']
        )
        assert np.isclose(result['confidence'], expected_confidence)

    def test_risk_level_values(self, temp_model_file, valid_batch_samples):
        """Test that risk levels are valid categorical values"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        result = predictor.predict_batch(valid_batch_samples, validate=False)

        valid_risk_levels = ['Low', 'Moderate', 'High']
        assert all(level in valid_risk_levels for level in result['risk_level'])

    def test_risk_level_thresholds(self, temp_model_file):
        """Test that risk levels correspond to correct probability thresholds"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        # Create samples with known probabilities (mocked)
        sample = {
            'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }

        result = predictor.predict_single(sample, validate=False)
        prob = result['probability_positive']
        risk = result['risk_level']

        # Verify risk level matches probability
        if prob < 0.3:
            assert risk == 'Low'
        elif prob < 0.6:
            assert risk == 'Moderate'
        else:
            assert risk == 'High'


class TestInputValidation:
    """Test cases for input validation"""

    def test_validate_complete_valid_input(self, temp_model_file, valid_single_sample):
        """Test validation passes for complete valid input"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        is_valid, errors = predictor.validate_input(valid_single_sample)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_features(self, temp_model_file):
        """Test validation fails for missing required features"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        incomplete_data = {
            'age': 63,
            'sex': 1,
            'cp': 3
        }

        is_valid, errors = predictor.validate_input(incomplete_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any('Missing required features' in error for error in errors)

    def test_validate_null_values(self, temp_model_file, valid_single_sample):
        """Test validation fails for null values"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        data_with_null = valid_single_sample.copy()
        data_with_null['age'] = None

        is_valid, errors = predictor.validate_input(data_with_null)

        assert is_valid is False
        assert len(errors) > 0
        assert any('null' in error.lower() for error in errors)

    def test_validate_out_of_range_values(self, temp_model_file):
        """Test validation fails for out-of-range values"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        invalid_data = {
            'age': 150,  # Outside valid range
            'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }

        is_valid, errors = predictor.validate_input(invalid_data)

        assert is_valid is False
        assert len(errors) > 0
        assert any('range' in error.lower() for error in errors)

    def test_validate_multiple_out_of_range(self, temp_model_file):
        """Test validation catches multiple out-of-range features"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        invalid_data = {
            'age': 150,  # Out of range
            'sex': 5,    # Out of range
            'cp': 10,    # Out of range
            'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }

        is_valid, errors = predictor.validate_input(invalid_data)

        assert is_valid is False
        assert len(errors) >= 3  # At least 3 features out of range

    def test_validate_negative_values_where_invalid(self, temp_model_file):
        """Test validation fails for negative values in features that don't allow them"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        invalid_data = {
            'age': -5,  # Negative age
            'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }

        is_valid, errors = predictor.validate_input(invalid_data)

        assert is_valid is False

    def test_validate_dataframe_input(self, temp_model_file, valid_batch_samples):
        """Test validation works with DataFrame input"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        is_valid, errors = predictor.validate_input(valid_batch_samples)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_batch_with_some_invalid(self, temp_model_file):
        """Test validation detects invalid rows in batch"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        batch = pd.DataFrame([
            {
                'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
                'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
                'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
            },
            {
                'age': 200,  # Invalid
                'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
                'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
                'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
            }
        ])

        is_valid, errors = predictor.validate_input(batch)
        assert is_valid is False

    def test_prediction_with_validation_enabled(self, temp_model_file, valid_single_sample):
        """Test that prediction with validation enabled validates input"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        # Should succeed with valid input
        result = predictor.predict_single(valid_single_sample, validate=True)
        assert 'prediction' in result

    def test_prediction_with_validation_fails_invalid_input(self, temp_model_file):
        """Test that prediction with validation fails on invalid input"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        invalid_data = {
            'age': 200,  # Invalid
            'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
            'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
        }

        with pytest.raises(ValueError) as exc_info:
            predictor.predict_single(invalid_data, validate=True)

        assert "validation failed" in str(exc_info.value).lower()


class TestPreprocessingConsistency:
    """Test cases for preprocessing consistency"""

    def test_preprocess_dict_to_dataframe(self, temp_model_file, valid_single_sample):
        """Test preprocessing converts dict to DataFrame correctly"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        processed = predictor.preprocess_input(valid_single_sample)

        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == 1

    def test_preprocess_maintains_feature_order(self, temp_model_file, valid_single_sample):
        """Test that preprocessing maintains correct feature order"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        # Provide features in wrong order
        unordered_sample = {k: valid_single_sample[k] for k in reversed(FEATURE_NAMES)}
        processed = predictor.preprocess_input(unordered_sample)

        assert list(processed.columns) == FEATURE_NAMES

    def test_preprocess_dataframe_unchanged_order(self, temp_model_file, valid_batch_samples):
        """Test preprocessing maintains feature order for DataFrames"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        processed = predictor.preprocess_input(valid_batch_samples)

        assert list(processed.columns) == FEATURE_NAMES

    def test_preprocess_selects_only_required_features(self, temp_model_file, valid_single_sample):
        """Test preprocessing selects only required features"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        # Add extra features
        data_with_extra = valid_single_sample.copy()
        data_with_extra['extra_feature1'] = 999
        data_with_extra['extra_feature2'] = 'test'

        processed = predictor.preprocess_input(data_with_extra)

        assert list(processed.columns) == FEATURE_NAMES
        assert 'extra_feature1' not in processed.columns
        assert 'extra_feature2' not in processed.columns

    def test_preprocess_data_types(self, temp_model_file, valid_single_sample):
        """Test that preprocessing converts to correct data types"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        processed = predictor.preprocess_input(valid_single_sample)

        # Float features
        float_features = ['age', 'trestbps', 'chol', 'thalach']
        for feature in float_features:
            assert processed[feature].dtype in [np.float64, np.float32, float]

        # Integer features
        int_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for feature in int_features:
            assert processed[feature].dtype in [np.int64, np.int32, int]

    def test_preprocess_consistent_across_calls(self, temp_model_file, valid_single_sample):
        """Test that preprocessing produces consistent results"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        processed1 = predictor.preprocess_input(valid_single_sample)
        processed2 = predictor.preprocess_input(valid_single_sample)

        pd.testing.assert_frame_equal(processed1, processed2)

    def test_preprocess_batch_preserves_order(self, temp_model_file, valid_batch_samples):
        """Test that batch preprocessing preserves row order"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)
        processed = predictor.preprocess_input(valid_batch_samples)

        # Check first and last row to ensure order preserved
        assert processed.iloc[0]['age'] == valid_batch_samples.iloc[0]['age']
        assert processed.iloc[-1]['age'] == valid_batch_samples.iloc[-1]['age']

    def test_preprocess_pipeline_consistency(self, temp_model_file, valid_single_sample):
        """Test that preprocessing followed by prediction is consistent"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        # Preprocess and predict multiple times
        results = []
        for _ in range(5):
            processed = predictor.preprocess_input(valid_single_sample)
            pred = predictor.pipeline.predict(processed)
            results.append(pred[0])

        # All predictions should be identical
        assert all(r == results[0] for r in results)

    def test_preprocessing_handles_edge_values(self, temp_model_file):
        """Test preprocessing handles edge case values correctly"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        # Use minimum valid values
        min_sample = {
            'age': 29, 'sex': 0, 'cp': 0, 'trestbps': 94, 'chol': 126,
            'fbs': 0, 'restecg': 0, 'thalach': 71, 'exang': 0,
            'oldpeak': 0.0, 'slope': 0, 'ca': 0, 'thal': 0
        }

        processed = predictor.preprocess_input(min_sample)
        assert len(processed) == 1
        assert all(processed.iloc[0][feature] == min_sample[feature] for feature in FEATURE_NAMES)

    def test_preprocessing_preserves_numeric_precision(self, temp_model_file):
        """Test that preprocessing preserves numeric precision"""
        predictor = HeartDiseasePredictor(model_path=temp_model_file)

        sample = {
            'age': 63.5, 'sex': 1, 'cp': 3, 'trestbps': 145.7, 'chol': 233.3,
            'fbs': 1, 'restecg': 0, 'thalach': 150.2, 'exang': 0,
            'oldpeak': 2.345, 'slope': 0, 'ca': 0, 'thal': 1
        }

        processed = predictor.preprocess_input(sample)

        # Check float precision is maintained
        assert np.isclose(processed.iloc[0]['oldpeak'], 2.345)


class TestFeatureRangeConstants:
    """Test cases for feature range constants"""

    def test_feature_ranges_exist(self):
        """Test that feature ranges are defined for all features"""
        assert hasattr(HeartDiseasePredictor, 'FEATURE_RANGES')
        assert isinstance(HeartDiseasePredictor.FEATURE_RANGES, dict)

    def test_all_features_have_ranges(self):
        """Test that all features have defined ranges"""
        ranges = HeartDiseasePredictor.FEATURE_RANGES

        for feature in FEATURE_NAMES:
            assert feature in ranges
            assert isinstance(ranges[feature], tuple)
            assert len(ranges[feature]) == 2

    def test_feature_ranges_valid(self):
        """Test that feature ranges are logically valid (min < max)"""
        ranges = HeartDiseasePredictor.FEATURE_RANGES

        for feature, (min_val, max_val) in ranges.items():
            assert min_val <= max_val, f"Invalid range for {feature}: {min_val} > {max_val}"

    def test_feature_descriptions_exist(self):
        """Test that feature descriptions are defined"""
        assert hasattr(HeartDiseasePredictor, 'FEATURE_DESCRIPTIONS')
        assert isinstance(HeartDiseasePredictor.FEATURE_DESCRIPTIONS, dict)

        for feature in FEATURE_NAMES:
            assert feature in HeartDiseasePredictor.FEATURE_DESCRIPTIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
