"""
Prediction Examples - Demonstrating various ways to use the predictor

Author: Umang Sharma (2024AC05070)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predict import HeartDiseasePredictor
import pandas as pd
import json


def example_single_prediction():
    """Example 1: Single prediction with dictionary input"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Prediction")
    print("="*80)

    # Initialize predictor (auto-detects best model)
    predictor = HeartDiseasePredictor()

    # Sample patient data
    patient_data = {
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

    # Make prediction
    result = predictor.predict_single(patient_data)

    # Display results
    print(f"\nPrediction: {result['prediction_label']}")
    print(f"Probability (No Disease): {result['probability_negative']:.4f}")
    print(f"Probability (Disease): {result['probability_positive']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']:.4f}")


def example_batch_prediction():
    """Example 2: Batch prediction with DataFrame"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction")
    print("="*80)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Sample batch data
    batch_data = pd.DataFrame([
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

    # Make predictions
    results = predictor.predict_batch(batch_data)

    # Display results
    print(f"\nProcessed {len(results)} samples")
    print("\nResults:")
    print(results[['age', 'prediction_label', 'probability_positive', 'risk_level']])


def example_raw_prediction():
    """Example 3: Raw predictions without formatting"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Raw Predictions")
    print("="*80)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Sample data
    sample = {
        'age': 56, 'sex': 1, 'cp': 1, 'trestbps': 120, 'chol': 236,
        'fbs': 0, 'restecg': 1, 'thalach': 178, 'exang': 0,
        'oldpeak': 0.8, 'slope': 2, 'ca': 0, 'thal': 2
    }

    # Get raw predictions and probabilities
    predictions, probabilities = predictor.predict(sample, return_proba=True)

    print(f"\nRaw Prediction: {predictions[0]}")
    print(f"Raw Probabilities: {probabilities[0]}")
    print(f"Class 0 (No Disease): {probabilities[0][0]:.4f}")
    print(f"Class 1 (Disease): {probabilities[0][1]:.4f}")


def example_validation():
    """Example 4: Input validation"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Input Validation")
    print("="*80)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Invalid sample (age out of range)
    invalid_sample = {
        'age': 120,  # Invalid age
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

    # Validate input
    is_valid, errors = predictor.validate_input(invalid_sample)

    if is_valid:
        print("\nInput is valid!")
    else:
        print("\nValidation failed!")
        print("Errors:")
        for error in errors:
            print(f"  - {error}")


def example_feature_info():
    """Example 5: Get feature information"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Feature Information")
    print("="*80)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Get feature information
    feature_info = predictor.get_feature_info()

    print("\nFeature Details:")
    print(feature_info.to_string(index=False))


def example_model_info():
    """Example 6: Get model information"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Model Information")
    print("="*80)

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Get model information
    model_info = predictor.get_model_info()

    print("\nModel Details:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")


def example_api_integration():
    """Example 7: API integration pattern"""
    print("\n" + "="*80)
    print("EXAMPLE 7: API Integration Pattern")
    print("="*80)

    # Initialize predictor (do this once at application startup)
    predictor = HeartDiseasePredictor()

    # Simulate API request data
    api_request = {
        'age': 45,
        'sex': 0,
        'cp': 1,
        'trestbps': 130,
        'chol': 234,
        'fbs': 0,
        'restecg': 0,
        'thalach': 175,
        'exang': 0,
        'oldpeak': 0.6,
        'slope': 1,
        'ca': 0,
        'thal': 2
    }

    try:
        # Make prediction
        result = predictor.predict_single(api_request)

        # Format API response
        api_response = {
            'success': True,
            'data': {
                'prediction': result['prediction_label'],
                'risk_level': result['risk_level'],
                'confidence': round(result['confidence'], 4),
                'probabilities': {
                    'no_disease': round(result['probability_negative'], 4),
                    'disease': round(result['probability_positive'], 4)
                }
            }
        }

        print("\nAPI Response:")
        print(json.dumps(api_response, indent=2))

    except Exception as e:
        # Error handling for API
        api_response = {
            'success': False,
            'error': str(e)
        }
        print("\nAPI Error Response:")
        print(json.dumps(api_response, indent=2))


def example_file_based_prediction():
    """Example 8: File-based prediction"""
    print("\n" + "="*80)
    print("EXAMPLE 8: File-based Prediction")
    print("="*80)

    from src.models.predict import load_input_from_file

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Load from JSON file
    try:
        json_data = load_input_from_file('data/sample_input.json')
        result = predictor.predict_single(json_data)
        print("\nJSON file prediction:")
        print(f"  Prediction: {result['prediction_label']}")
        print(f"  Risk Level: {result['risk_level']}")
    except FileNotFoundError:
        print("\nJSON file not found (run from project root)")

    # Load from CSV file
    try:
        csv_data = load_input_from_file('data/sample_batch.csv')
        results = predictor.predict_batch(csv_data)
        print(f"\nCSV file prediction:")
        print(f"  Processed {len(results)} samples")
        print(f"  Positive predictions: {(results['prediction'] == 1).sum()}")
    except FileNotFoundError:
        print("\nCSV file not found (run from project root)")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("HEART DISEASE PREDICTION EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate various ways to use the HeartDiseasePredictor")

    try:
        example_single_prediction()
        example_batch_prediction()
        example_raw_prediction()
        example_validation()
        example_feature_info()
        example_model_info()
        example_api_integration()
        example_file_based_prediction()

        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED")
        print("="*80 + "\n")

    except FileNotFoundError as e:
        print(f"\n\nError: {e}")
        print("\nPlease train a model first:")
        print("  python src/models/train.py")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
