"""
Prediction Module for Heart Disease Classification

This module provides functionality to load trained models and make predictions
on new data. It includes input validation, preprocessing, and can be used
both programmatically and via CLI.

Author: Umang Sharma (2024AC05070)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import FEATURE_NAMES, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """
    A class to handle predictions for heart disease classification.

    This class loads trained models and preprocessing pipelines to make
    predictions on new patient data. It supports both single sample and
    batch predictions.

    Attributes:
        model_path (Path): Path to the trained model file.
        pipeline (Pipeline): Loaded sklearn pipeline with preprocessor and model.
        feature_names (List[str]): List of expected feature names.
        model_info (Dict): Information about the loaded model.
    """

    # Feature value ranges for validation
    FEATURE_RANGES = {
        'age': (29, 77),
        'sex': (0, 1),
        'cp': (0, 3),
        'trestbps': (94, 200),
        'chol': (126, 564),
        'fbs': (0, 1),
        'restecg': (0, 2),
        'thalach': (71, 202),
        'exang': (0, 1),
        'oldpeak': (0.0, 6.2),
        'slope': (0, 2),
        'ca': (0, 4),
        'thal': (0, 3)
    }

    FEATURE_DESCRIPTIONS = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-4)',
        'thal': 'Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect; 3 = unknown)'
    }

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path: Path to the trained model pipeline. If None, attempts to
                       load the default best model from the models directory.

        Raises:
            FileNotFoundError: If model file is not found.
            Exception: If model loading fails.
        """
        self.feature_names = FEATURE_NAMES
        self.pipeline = None
        self.model_info = {}

        # Determine model path
        if model_path is None:
            model_path = self._find_best_model()
        else:
            model_path = Path(model_path)

        self.model_path = model_path

        # Load the model
        self._load_model()

        logger.info(f"Predictor initialized with model: {self.model_path}")

    def _find_best_model(self) -> Path:
        """
        Find the best model in the models directory.

        Returns:
            Path to the best model file.

        Raises:
            FileNotFoundError: If no models are found.
        """
        models_dir = Path(MODELS_DIR)

        # Look for pipeline files (preferred)
        pipeline_files = list(models_dir.glob("*_pipeline.pkl"))

        if pipeline_files:
            # Check for model comparison file to find best model
            comparison_file = models_dir / "model_comparison.csv"
            if comparison_file.exists():
                try:
                    comparison_df = pd.read_csv(comparison_file)
                    best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
                    best_model_path = models_dir / f"{best_model_name}_pipeline.pkl"

                    if best_model_path.exists():
                        logger.info(f"Found best model: {best_model_name}")
                        return best_model_path
                except Exception as e:
                    logger.warning(f"Could not read comparison file: {e}")

            # If comparison file not found, use the first pipeline
            logger.warning("Using first available pipeline model")
            return pipeline_files[0]

        raise FileNotFoundError(
            f"No trained models found in {models_dir}. "
            "Please train a model first using: python src/models/train.py"
        )

    def _load_model(self):
        """
        Load the trained model pipeline from disk.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            Exception: If model loading fails.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please train a model first using: python src/models/train.py"
            )

        try:
            self.pipeline = joblib.load(self.model_path)

            # Extract model information
            self.model_info = {
                'model_path': str(self.model_path),
                'model_type': type(self.pipeline.named_steps['classifier']).__name__,
                'has_scaler': 'scaler' in self.pipeline.named_steps,
                'n_features': len(self.feature_names)
            }

            logger.info(f"Model loaded successfully: {self.model_info['model_type']}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def validate_input(self, data: Union[Dict, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate input data against expected schema and ranges.

        Args:
            data: Input data as dictionary or DataFrame.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []

        # Convert to DataFrame if dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Check for missing features
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            errors.append(f"Missing required features: {list(missing_features)}")
            return False, errors

        # Check for extra features (warning only)
        extra_features = set(df.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features will be ignored: {list(extra_features)}")

        # Validate feature ranges
        for feature in self.feature_names:
            if feature in df.columns:
                values = df[feature]

                # Check for null values
                if values.isnull().any():
                    errors.append(f"Feature '{feature}' contains null values")
                    continue

                # Check ranges
                if feature in self.FEATURE_RANGES:
                    min_val, max_val = self.FEATURE_RANGES[feature]
                    out_of_range = (values < min_val) | (values > max_val)

                    if out_of_range.any():
                        invalid_values = values[out_of_range].unique()
                        errors.append(
                            f"Feature '{feature}' has values outside valid range "
                            f"[{min_val}, {max_val}]: {list(invalid_values)}"
                        )

        is_valid = len(errors) == 0
        return is_valid, errors

    def preprocess_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data to match model expectations.

        Args:
            data: Input data as dictionary or DataFrame.

        Returns:
            Preprocessed DataFrame with correct feature order.
        """
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Select and order features
        df = df[self.feature_names]

        # Ensure correct data types
        for feature in self.feature_names:
            if feature in ['age', 'trestbps', 'chol', 'thalach']:
                df[feature] = df[feature].astype(float)
            else:
                df[feature] = df[feature].astype(int)

        return df

    def predict(
        self,
        data: Union[Dict, pd.DataFrame],
        return_proba: bool = True,
        validate: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on input data.

        Args:
            data: Input data as dictionary (single sample) or DataFrame (batch).
            return_proba: If True, return probability scores along with predictions.
            validate: If True, validate input data before prediction.

        Returns:
            If return_proba is True: Tuple of (predictions, probabilities)
            If return_proba is False: predictions only

        Raises:
            ValueError: If input validation fails.
            Exception: If prediction fails.
        """
        try:
            # Validate input
            if validate:
                is_valid, errors = self.validate_input(data)
                if not is_valid:
                    error_msg = "Input validation failed:\n" + "\n".join(errors)
                    raise ValueError(error_msg)

            # Preprocess input
            X = self.preprocess_input(data)

            # Make predictions
            predictions = self.pipeline.predict(X)

            if return_proba:
                probabilities = self.pipeline.predict_proba(X)
                return predictions, probabilities

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_single(
        self,
        sample: Dict,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction for a single sample with formatted output.

        Args:
            sample: Dictionary containing feature values.
            validate: If True, validate input data.

        Returns:
            Dictionary containing prediction results with probabilities and interpretation.
        """
        predictions, probabilities = self.predict(
            sample,
            return_proba=True,
            validate=validate
        )

        prediction = int(predictions[0])
        prob_negative = float(probabilities[0][0])
        prob_positive = float(probabilities[0][1])

        # Determine risk level
        if prob_positive < 0.3:
            risk_level = "Low"
        elif prob_positive < 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        result = {
            'prediction': prediction,
            'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'probability_positive': prob_positive,
            'probability_negative': prob_negative,
            'risk_level': risk_level,
            'confidence': max(prob_positive, prob_negative),
            'input_features': sample
        }

        return result

    def predict_batch(
        self,
        data: pd.DataFrame,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of samples.

        Args:
            data: DataFrame containing multiple samples.
            validate: If True, validate input data.

        Returns:
            DataFrame with predictions and probabilities.
        """
        predictions, probabilities = self.predict(
            data,
            return_proba=True,
            validate=validate
        )

        results_df = data.copy()
        results_df['prediction'] = predictions
        results_df['prediction_label'] = results_df['prediction'].map({
            0: 'No Heart Disease',
            1: 'Heart Disease'
        })
        results_df['probability_negative'] = probabilities[:, 0]
        results_df['probability_positive'] = probabilities[:, 1]
        results_df['confidence'] = np.max(probabilities, axis=1)

        # Add risk level
        results_df['risk_level'] = pd.cut(
            results_df['probability_positive'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Moderate', 'High']
        )

        return results_df

    def get_feature_info(self) -> pd.DataFrame:
        """
        Get information about expected features.

        Returns:
            DataFrame containing feature information.
        """
        feature_info = []
        for feature in self.feature_names:
            info = {
                'feature': feature,
                'description': self.FEATURE_DESCRIPTIONS.get(feature, 'N/A')
            }

            if feature in self.FEATURE_RANGES:
                min_val, max_val = self.FEATURE_RANGES[feature]
                info['min'] = min_val
                info['max'] = max_val

            feature_info.append(info)

        return pd.DataFrame(feature_info)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information.
        """
        return self.model_info


def load_input_from_file(file_path: str) -> Union[Dict, pd.DataFrame]:
    """
    Load input data from a file (JSON or CSV).

    Args:
        file_path: Path to the input file.

    Returns:
        Loaded data as dictionary or DataFrame.

    Raises:
        ValueError: If file format is not supported.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    elif file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .json or .csv")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Make predictions using trained heart disease model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show feature information
  python src/models/predict.py --features

  # Predict from JSON file
  python src/models/predict.py --input sample.json

  # Predict from CSV file (batch)
  python src/models/predict.py --input data.csv --output predictions.csv

  # Predict single sample via CLI
  python src/models/predict.py --age 63 --sex 1 --cp 3 --trestbps 145 --chol 233 \\
    --fbs 1 --restecg 0 --thalach 150 --exang 0 --oldpeak 2.3 --slope 0 --ca 0 --thal 1
        """
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model pipeline (default: auto-detect best model)'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input file path (JSON for single sample, CSV for batch)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for batch predictions (CSV format)'
    )

    parser.add_argument(
        '--features',
        action='store_true',
        help='Display feature information and exit'
    )

    parser.add_argument(
        '--model-info',
        action='store_true',
        help='Display model information and exit'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip input validation'
    )

    # Individual feature arguments for CLI input
    for feature in FEATURE_NAMES:
        parser.add_argument(
            f'--{feature}',
            type=float,
            help=f'{HeartDiseasePredictor.FEATURE_DESCRIPTIONS.get(feature, feature)}'
        )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def main():
    """
    Main function for CLI interface.
    """
    args = parse_arguments()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize predictor
        predictor = HeartDiseasePredictor(model_path=args.model_path)

        # Display feature information
        if args.features:
            feature_info = predictor.get_feature_info()
            print("\n" + "="*80)
            print("FEATURE INFORMATION")
            print("="*80)
            print(feature_info.to_string(index=False))
            print("\n")
            return 0

        # Display model information
        if args.model_info:
            model_info = predictor.get_model_info()
            print("\n" + "="*80)
            print("MODEL INFORMATION")
            print("="*80)
            for key, value in model_info.items():
                print(f"{key}: {value}")
            print("\n")
            return 0

        # Load input from file
        if args.input:
            data = load_input_from_file(args.input)

            # Batch prediction (CSV)
            if isinstance(data, pd.DataFrame):
                logger.info(f"Processing batch of {len(data)} samples...")
                results = predictor.predict_batch(
                    data,
                    validate=not args.no_validate
                )

                print("\n" + "="*80)
                print("BATCH PREDICTION RESULTS")
                print("="*80)
                print(f"Total samples: {len(results)}")
                print(f"Predicted positive: {(results['prediction'] == 1).sum()}")
                print(f"Predicted negative: {(results['prediction'] == 0).sum()}")
                print("\nSample results:")
                print(results[['prediction_label', 'probability_positive', 'risk_level']].head(10))

                # Save results if output path provided
                if args.output:
                    results.to_csv(args.output, index=False)
                    logger.info(f"Results saved to: {args.output}")

            # Single prediction (JSON)
            else:
                result = predictor.predict_single(
                    data,
                    validate=not args.no_validate
                )

                print("\n" + "="*80)
                print("PREDICTION RESULT")
                print("="*80)
                print(f"Prediction: {result['prediction_label']}")
                print(f"Probability (No Disease): {result['probability_negative']:.4f}")
                print(f"Probability (Disease): {result['probability_positive']:.4f}")
                print(f"Risk Level: {result['risk_level']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("\n")

        # CLI input (individual features)
        else:
            # Check if any features provided
            feature_values = {f: getattr(args, f) for f in FEATURE_NAMES}
            if all(v is None for v in feature_values.values()):
                logger.error("No input provided. Use --input file or provide feature values via CLI.")
                logger.info("Run with --help for usage examples.")
                return 1

            # Check if all features provided
            missing = [f for f, v in feature_values.items() if v is None]
            if missing:
                logger.error(f"Missing required features: {missing}")
                logger.info("Run with --features to see all required features.")
                return 1

            # Make prediction
            result = predictor.predict_single(
                feature_values,
                validate=not args.no_validate
            )

            print("\n" + "="*80)
            print("PREDICTION RESULT")
            print("="*80)
            print(f"Prediction: {result['prediction_label']}")
            print(f"Probability (No Disease): {result['probability_negative']:.4f}")
            print(f"Probability (Disease): {result['probability_positive']:.4f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("\n")

        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=args.log_level == 'DEBUG')
        return 1


if __name__ == "__main__":
    sys.exit(main())
