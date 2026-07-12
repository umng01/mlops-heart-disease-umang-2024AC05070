"""
Comprehensive API tests for the Heart Disease Prediction API.

This module tests:
- Health check endpoint
- Prediction endpoint with valid input
- Prediction endpoint with invalid input
- Response schema validation
- Error handling
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.app import app, load_model, get_risk_level


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


@pytest.fixture
def valid_patient_data():
    """Valid patient data for testing."""
    return {
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
    }


@pytest.fixture
def setup_model(mock_model):
    """Setup mock model in the app."""
    with patch('api.app.MODEL', mock_model):
        yield mock_model


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_success(self, client):
        """Test health check returns correct response."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

        # Validate response values
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
        assert data["version"] == "1.0.0"

    def test_health_check_model_not_loaded(self, client):
        """Test health check when model is not loaded."""
        with patch('api.app.MODEL', None):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is False

    def test_health_check_model_loaded(self, client, setup_model):
        """Test health check when model is loaded."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True

    def test_health_check_response_schema(self, client):
        """Test health check response matches expected schema."""
        response = client.get("/health")
        data = response.json()

        # Check all required fields are present
        required_fields = ["status", "model_loaded", "version"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Check field types
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["version"], str)


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_valid_input(self, client, valid_patient_data, setup_model):
        """Test prediction with valid input."""
        response = client.post("/predict", json=valid_patient_data)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data

        # Validate data types
        assert isinstance(data["prediction"], int)
        assert isinstance(data["probability"], (int, float))
        assert isinstance(data["risk_level"], str)

        # Validate value ranges
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["low", "moderate", "high"]

    def test_predict_high_risk(self, client, valid_patient_data):
        """Test prediction returns high risk for high probability."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

        with patch('api.app.MODEL', mock_model):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 200
            data = response.json()
            assert data["risk_level"] == "high"
            assert data["prediction"] == 1

    def test_predict_low_risk(self, client, valid_patient_data):
        """Test prediction returns low risk for low probability."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

        with patch('api.app.MODEL', mock_model):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 200
            data = response.json()
            assert data["risk_level"] == "low"
            assert data["prediction"] == 0

    def test_predict_moderate_risk(self, client, valid_patient_data):
        """Test prediction returns moderate risk for medium probability."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.5, 0.5]])

        with patch('api.app.MODEL', mock_model):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 200
            data = response.json()
            assert data["risk_level"] == "moderate"

    def test_predict_model_not_loaded(self, client, valid_patient_data):
        """Test prediction fails when model is not loaded."""
        with patch('api.app.MODEL', None):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 503
            data = response.json()
            assert "detail" in data
            assert "model not loaded" in data["detail"].lower()

    def test_predict_probability_rounded(self, client, valid_patient_data):
        """Test that probability is rounded to 4 decimal places."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.123456789, 0.876543211]])

        with patch('api.app.MODEL', mock_model):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 200
            data = response.json()

            # Check probability is rounded to 4 decimal places
            probability_str = str(data["probability"])
            decimal_places = len(probability_str.split('.')[-1]) if '.' in probability_str else 0
            assert decimal_places <= 4


class TestInvalidInput:
    """Tests for invalid input handling."""

    def test_predict_missing_required_field(self, client, valid_patient_data):
        """Test prediction with missing required field."""
        invalid_data = valid_patient_data.copy()
        del invalid_data["age"]

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_age_negative(self, client, valid_patient_data):
        """Test prediction with negative age."""
        invalid_data = valid_patient_data.copy()
        invalid_data["age"] = -5

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_age_too_high(self, client, valid_patient_data):
        """Test prediction with age over 120."""
        invalid_data = valid_patient_data.copy()
        invalid_data["age"] = 150

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_sex(self, client, valid_patient_data):
        """Test prediction with invalid sex value."""
        invalid_data = valid_patient_data.copy()
        invalid_data["sex"] = 2

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_cp(self, client, valid_patient_data):
        """Test prediction with invalid chest pain type."""
        invalid_data = valid_patient_data.copy()
        invalid_data["cp"] = 5

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_trestbps_negative(self, client, valid_patient_data):
        """Test prediction with negative blood pressure."""
        invalid_data = valid_patient_data.copy()
        invalid_data["trestbps"] = -10

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_trestbps_too_high(self, client, valid_patient_data):
        """Test prediction with blood pressure too high."""
        invalid_data = valid_patient_data.copy()
        invalid_data["trestbps"] = 350

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_chol_negative(self, client, valid_patient_data):
        """Test prediction with negative cholesterol."""
        invalid_data = valid_patient_data.copy()
        invalid_data["chol"] = -50

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_chol_too_high(self, client, valid_patient_data):
        """Test prediction with cholesterol too high."""
        invalid_data = valid_patient_data.copy()
        invalid_data["chol"] = 700

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_thalach_negative(self, client, valid_patient_data):
        """Test prediction with negative heart rate."""
        invalid_data = valid_patient_data.copy()
        invalid_data["thalach"] = -10

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_thalach_too_high(self, client, valid_patient_data):
        """Test prediction with heart rate too high."""
        invalid_data = valid_patient_data.copy()
        invalid_data["thalach"] = 300

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_oldpeak_negative(self, client, valid_patient_data):
        """Test prediction with negative oldpeak."""
        invalid_data = valid_patient_data.copy()
        invalid_data["oldpeak"] = -1.5

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_oldpeak_too_high(self, client, valid_patient_data):
        """Test prediction with oldpeak too high."""
        invalid_data = valid_patient_data.copy()
        invalid_data["oldpeak"] = 15.0

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_ca(self, client, valid_patient_data):
        """Test prediction with invalid ca value."""
        invalid_data = valid_patient_data.copy()
        invalid_data["ca"] = 5

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_thal(self, client, valid_patient_data):
        """Test prediction with invalid thal value."""
        invalid_data = valid_patient_data.copy()
        invalid_data["thal"] = 5

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_null_value(self, client, valid_patient_data):
        """Test prediction with null value."""
        invalid_data = valid_patient_data.copy()
        invalid_data["age"] = None

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_wrong_type_string(self, client, valid_patient_data):
        """Test prediction with string instead of integer."""
        invalid_data = valid_patient_data.copy()
        invalid_data["age"] = "not_a_number"

        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_empty_body(self, client):
        """Test prediction with empty request body."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_extra_fields(self, client, valid_patient_data, setup_model):
        """Test prediction with extra fields (should be ignored)."""
        invalid_data = valid_patient_data.copy()
        invalid_data["extra_field"] = "should_be_ignored"

        response = client.post("/predict", json=invalid_data)
        # FastAPI/Pydantic ignores extra fields by default
        assert response.status_code == 200


class TestResponseSchema:
    """Tests for response schema validation."""

    def test_prediction_response_schema(self, client, valid_patient_data, setup_model):
        """Test prediction response has correct schema."""
        response = client.post("/predict", json=valid_patient_data)

        assert response.status_code == 200
        data = response.json()

        # Required fields
        required_fields = ["prediction", "probability", "risk_level"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # No extra fields
        for field in data.keys():
            assert field in required_fields, f"Unexpected field: {field}"

    def test_health_response_schema(self, client):
        """Test health response has correct schema."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        required_fields = ["status", "model_loaded", "version"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_error_response_schema(self, client, valid_patient_data):
        """Test error response has correct schema."""
        with patch('api.app.MODEL', None):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 503
            data = response.json()
            assert "detail" in data


class TestErrorHandling:
    """Tests for error handling."""

    def test_prediction_error_handling(self, client, valid_patient_data):
        """Test prediction handles model errors gracefully."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model prediction failed")

        with patch('api.app.MODEL', mock_model):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    def test_prediction_value_error(self, client, valid_patient_data):
        """Test prediction handles ValueError."""
        mock_model = Mock()
        mock_model.predict.side_effect = ValueError("Invalid input")

        with patch('api.app.MODEL', mock_model):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 500

    def test_model_without_predict_proba(self, client, valid_patient_data):
        """Test prediction with model that doesn't have predict_proba."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        # Remove predict_proba method
        delattr(mock_model, 'predict_proba')
        mock_model.decision_function.return_value = np.array([2.5])

        with patch('api.app.MODEL', mock_model):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 200
            data = response.json()
            assert "probability" in data

    def test_model_without_predict_proba_or_decision_function(self, client, valid_patient_data):
        """Test prediction with model that has neither predict_proba nor decision_function."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        # Remove both methods
        delattr(mock_model, 'predict_proba')
        delattr(mock_model, 'decision_function')

        with patch('api.app.MODEL', mock_model):
            response = client.post("/predict", json=valid_patient_data)

            assert response.status_code == 200
            data = response.json()
            assert data["probability"] == float(data["prediction"])


class TestRiskLevelFunction:
    """Tests for the get_risk_level utility function."""

    def test_risk_level_low(self):
        """Test low risk level."""
        assert get_risk_level(0.1) == "low"
        assert get_risk_level(0.29) == "low"

    def test_risk_level_moderate(self):
        """Test moderate risk level."""
        assert get_risk_level(0.3) == "moderate"
        assert get_risk_level(0.5) == "moderate"
        assert get_risk_level(0.69) == "moderate"

    def test_risk_level_high(self):
        """Test high risk level."""
        assert get_risk_level(0.7) == "high"
        assert get_risk_level(0.9) == "high"
        assert get_risk_level(1.0) == "high"

    def test_risk_level_boundaries(self):
        """Test risk level boundaries."""
        assert get_risk_level(0.0) == "low"
        assert get_risk_level(0.299) == "low"
        assert get_risk_level(0.301) == "moderate"
        assert get_risk_level(0.699) == "moderate"
        assert get_risk_level(0.701) == "high"


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_endpoint_exists(self, client):
        """Test metrics endpoint is accessible."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        """Test metrics endpoint returns correct content type."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_predict_minimum_values(self, client, setup_model):
        """Test prediction with minimum valid values."""
        min_data = {
            "age": 0,
            "sex": 0,
            "cp": 0,
            "trestbps": 0,
            "chol": 0,
            "fbs": 0,
            "restecg": 0,
            "thalach": 0,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 0,
            "ca": 0,
            "thal": 0
        }

        response = client.post("/predict", json=min_data)
        # Should work with minimum values
        assert response.status_code == 200

    def test_predict_maximum_values(self, client, setup_model):
        """Test prediction with maximum valid values."""
        max_data = {
            "age": 120,
            "sex": 1,
            "cp": 3,
            "trestbps": 300,
            "chol": 600,
            "fbs": 1,
            "restecg": 2,
            "thalach": 250,
            "exang": 1,
            "oldpeak": 10.0,
            "slope": 2,
            "ca": 4,
            "thal": 3
        }

        response = client.post("/predict", json=max_data)
        # Should work with maximum values
        assert response.status_code == 200

    def test_predict_float_for_integer_field(self, client, valid_patient_data, setup_model):
        """Test prediction with float value for integer field."""
        invalid_data = valid_patient_data.copy()
        invalid_data["age"] = 63.5

        # FastAPI/Pydantic will coerce float to int
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
