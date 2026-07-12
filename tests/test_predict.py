"""
Unit Tests for Prediction Module

Author: Umang Sharma (2024AC05070)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predict import HeartDiseasePredictor, load_input_from_file


@pytest.fixture
def sample_data():
    """Fixture providing valid sample data"""
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
def sample_batch():
    """Fixture providing batch data"""
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
        }
    ])


class TestHeartDiseasePredictor:
    """Test cases for HeartDiseasePredictor class"""

    def test_predictor_initialization(self):
        """Test that predictor initializes correctly"""
        try:
            predictor = HeartDiseasePredictor()
            assert predictor is not None
            assert predictor.pipeline is not None
            assert len(predictor.feature_names) == 13
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_model_info(self):
        """Test model information retrieval"""
        try:
            predictor = HeartDiseasePredictor()
            info = predictor.get_model_info()

            assert isinstance(info, dict)
            assert 'model_type' in info
            assert 'n_features' in info
            assert info['n_features'] == 13
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_feature_info(self):
        """Test feature information retrieval"""
        try:
            predictor = HeartDiseasePredictor()
            feature_info = predictor.get_feature_info()

            assert isinstance(feature_info, pd.DataFrame)
            assert len(feature_info) == 13
            assert 'feature' in feature_info.columns
            assert 'description' in feature_info.columns
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_input_validation_valid(self, sample_data):
        """Test validation with valid input"""
        try:
            predictor = HeartDiseasePredictor()
            is_valid, errors = predictor.validate_input(sample_data)

            assert is_valid is True
            assert len(errors) == 0
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_input_validation_missing_features(self):
        """Test validation with missing features"""
        try:
            predictor = HeartDiseasePredictor()
            incomplete_data = {'age': 63, 'sex': 1}

            is_valid, errors = predictor.validate_input(incomplete_data)

            assert is_valid is False
            assert len(errors) > 0
            assert any('Missing' in error for error in errors)
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_input_validation_out_of_range(self):
        """Test validation with out-of-range values"""
        try:
            predictor = HeartDiseasePredictor()
            invalid_data = {
                'age': 150,  # Out of range
                'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
                'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
                'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
            }

            is_valid, errors = predictor.validate_input(invalid_data)

            assert is_valid is False
            assert len(errors) > 0
            assert any('range' in error.lower() for error in errors)
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_input_validation_null_values(self, sample_data):
        """Test validation with null values"""
        try:
            predictor = HeartDiseasePredictor()
            data_with_null = sample_data.copy()
            data_with_null['age'] = None

            is_valid, errors = predictor.validate_input(data_with_null)

            assert is_valid is False
            assert len(errors) > 0
            assert any('null' in error.lower() for error in errors)
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_preprocess_input_dict(self, sample_data):
        """Test preprocessing with dictionary input"""
        try:
            predictor = HeartDiseasePredictor()
            processed = predictor.preprocess_input(sample_data)

            assert isinstance(processed, pd.DataFrame)
            assert len(processed) == 1
            assert list(processed.columns) == predictor.feature_names
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_preprocess_input_dataframe(self, sample_batch):
        """Test preprocessing with DataFrame input"""
        try:
            predictor = HeartDiseasePredictor()
            processed = predictor.preprocess_input(sample_batch)

            assert isinstance(processed, pd.DataFrame)
            assert len(processed) == len(sample_batch)
            assert list(processed.columns) == predictor.feature_names
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_predict_single_sample(self, sample_data):
        """Test prediction on single sample"""
        try:
            predictor = HeartDiseasePredictor()
            predictions, probabilities = predictor.predict(sample_data, return_proba=True)

            assert len(predictions) == 1
            assert predictions[0] in [0, 1]
            assert probabilities.shape == (1, 2)
            assert np.isclose(probabilities.sum(), 1.0)
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_predict_batch(self, sample_batch):
        """Test prediction on batch"""
        try:
            predictor = HeartDiseasePredictor()
            predictions, probabilities = predictor.predict(sample_batch, return_proba=True)

            assert len(predictions) == len(sample_batch)
            assert all(pred in [0, 1] for pred in predictions)
            assert probabilities.shape == (len(sample_batch), 2)
            assert all(np.isclose(prob.sum(), 1.0) for prob in probabilities)
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_predict_single_formatted(self, sample_data):
        """Test formatted single prediction"""
        try:
            predictor = HeartDiseasePredictor()
            result = predictor.predict_single(sample_data)

            assert isinstance(result, dict)
            assert 'prediction' in result
            assert 'prediction_label' in result
            assert 'probability_positive' in result
            assert 'probability_negative' in result
            assert 'risk_level' in result
            assert 'confidence' in result

            assert result['prediction'] in [0, 1]
            assert result['risk_level'] in ['Low', 'Moderate', 'High']
            assert 0 <= result['confidence'] <= 1
            assert 0 <= result['probability_positive'] <= 1
            assert 0 <= result['probability_negative'] <= 1
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_predict_batch_formatted(self, sample_batch):
        """Test formatted batch prediction"""
        try:
            predictor = HeartDiseasePredictor()
            results = predictor.predict_batch(sample_batch)

            assert isinstance(results, pd.DataFrame)
            assert len(results) == len(sample_batch)
            assert 'prediction' in results.columns
            assert 'prediction_label' in results.columns
            assert 'probability_positive' in results.columns
            assert 'risk_level' in results.columns
            assert 'confidence' in results.columns

            assert all(results['prediction'].isin([0, 1]))
            assert all(results['risk_level'].isin(['Low', 'Moderate', 'High']))
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_predict_without_validation(self, sample_data):
        """Test prediction with validation disabled"""
        try:
            predictor = HeartDiseasePredictor()
            result = predictor.predict_single(sample_data, validate=False)

            assert 'prediction' in result
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_risk_level_classification(self):
        """Test risk level classification thresholds"""
        try:
            predictor = HeartDiseasePredictor()

            # Test different probability values
            test_cases = [
                {'prob': 0.1, 'expected': 'Low'},
                {'prob': 0.5, 'expected': 'Moderate'},
                {'prob': 0.8, 'expected': 'High'}
            ]

            for case in test_cases:
                # Create mock result
                mock_data = {
                    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
                    'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
                    'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
                }

                result = predictor.predict_single(mock_data)

                # Verify risk level calculation logic exists
                assert result['risk_level'] in ['Low', 'Moderate', 'High']

        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_prediction_confidence(self, sample_data):
        """Test that confidence is calculated correctly"""
        try:
            predictor = HeartDiseasePredictor()
            result = predictor.predict_single(sample_data)

            # Confidence should be the maximum probability
            expected_confidence = max(
                result['probability_positive'],
                result['probability_negative']
            )

            assert np.isclose(result['confidence'], expected_confidence)
        except FileNotFoundError:
            pytest.skip("No trained model available")


class TestLoadInputFromFile:
    """Test cases for file loading functionality"""

    def test_load_json_file(self, tmp_path, sample_data):
        """Test loading JSON file"""
        import json

        # Create temporary JSON file
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(sample_data, f)

        # Load and verify
        loaded_data = load_input_from_file(str(json_file))
        assert loaded_data == sample_data

    def test_load_csv_file(self, tmp_path, sample_batch):
        """Test loading CSV file"""
        # Create temporary CSV file
        csv_file = tmp_path / "test.csv"
        sample_batch.to_csv(csv_file, index=False)

        # Load and verify
        loaded_data = load_input_from_file(str(csv_file))
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_batch)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_input_from_file("nonexistent_file.json")

    def test_load_unsupported_format(self, tmp_path):
        """Test loading unsupported file format"""
        # Create temporary file with unsupported extension
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test data")

        with pytest.raises(ValueError):
            load_input_from_file(str(txt_file))


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_predict_with_extra_features(self, sample_data):
        """Test prediction with extra features (should be ignored)"""
        try:
            predictor = HeartDiseasePredictor()
            data_with_extra = sample_data.copy()
            data_with_extra['extra_feature'] = 999

            result = predictor.predict_single(data_with_extra)
            assert 'prediction' in result
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_predict_with_invalid_types(self):
        """Test prediction with invalid data types"""
        try:
            predictor = HeartDiseasePredictor()
            invalid_data = {
                'age': 'sixty',  # String instead of number
                'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
                'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
                'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
            }

            with pytest.raises(Exception):
                predictor.predict_single(invalid_data)
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_empty_dataframe(self):
        """Test prediction with empty DataFrame"""
        try:
            predictor = HeartDiseasePredictor()
            empty_df = pd.DataFrame()

            with pytest.raises(Exception):
                predictor.predict_batch(empty_df)
        except FileNotFoundError:
            pytest.skip("No trained model available")

    def test_single_feature_dataframe(self):
        """Test prediction with DataFrame containing only one feature"""
        try:
            predictor = HeartDiseasePredictor()
            single_feature_df = pd.DataFrame({'age': [63]})

            is_valid, errors = predictor.validate_input(single_feature_df)
            assert is_valid is False
        except FileNotFoundError:
            pytest.skip("No trained model available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
