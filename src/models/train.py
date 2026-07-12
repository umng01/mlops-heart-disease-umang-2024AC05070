"""
Training script for heart disease prediction models.

This module provides functionality to train multiple machine learning models
with hyperparameter tuning, cross-validation, and MLflow tracking.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class to handle training of multiple ML models with MLflow tracking.

    Attributes:
        data_path (Path): Path to the preprocessed data directory.
        output_path (Path): Path to save trained models.
        experiment_name (str): MLflow experiment name.
        random_state (int): Random state for reproducibility.
    """

    def __init__(
        self,
        data_path: str,
        output_path: str,
        experiment_name: str = "heart_disease_prediction",
        random_state: int = 42
    ):
        """
        Initialize the ModelTrainer.

        Args:
            data_path: Path to the preprocessed data directory.
            output_path: Path to save trained models.
            experiment_name: Name for the MLflow experiment.
            random_state: Random state for reproducibility.
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.experiment_name = experiment_name
        self.random_state = random_state

        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow
        mlflow.set_experiment(self.experiment_name)

        logger.info(f"ModelTrainer initialized with data_path: {self.data_path}")
        logger.info(f"Models will be saved to: {self.output_path}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load preprocessed training and testing data.

        Returns:
            Tuple containing X_train, y_train, X_test, y_test.

        Raises:
            FileNotFoundError: If data files are not found.
            Exception: For other errors during data loading.
        """
        try:
            X_train_path = self.data_path / "X_train.csv"
            y_train_path = self.data_path / "y_train.csv"
            X_test_path = self.data_path / "X_test.csv"
            y_test_path = self.data_path / "y_test.csv"

            # Check if files exist
            for path in [X_train_path, y_train_path, X_test_path, y_test_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Data file not found: {path}")

            logger.info("Loading training and testing data...")
            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path).squeeze()
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).squeeze()

            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Testing data shape: {X_test.shape}")
            logger.info(f"Training labels distribution:\n{y_train.value_counts()}")

            return X_train, y_train, X_test, y_test

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model configurations with hyperparameter grids.

        Returns:
            Dictionary containing model configurations.
        """
        return {
            "logistic_regression": {
                "model": LogisticRegression(random_state=self.random_state, max_iter=1000),
                "param_grid": {
                    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "classifier__penalty": ["l1", "l2"],
                    "classifier__solver": ["liblinear", "saga"],
                }
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=self.random_state),
                "param_grid": {
                    "classifier__n_estimators": [50, 100, 200],
                    "classifier__max_depth": [None, 10, 20, 30],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4],
                    "classifier__max_features": ["sqrt", "log2"],
                }
            },
            "xgboost": {
                "model": xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric="logloss",
                    use_label_encoder=False
                ),
                "param_grid": {
                    "classifier__n_estimators": [50, 100, 200],
                    "classifier__max_depth": [3, 5, 7, 9],
                    "classifier__learning_rate": [0.01, 0.1, 0.3],
                    "classifier__subsample": [0.8, 0.9, 1.0],
                    "classifier__colsample_bytree": [0.8, 0.9, 1.0],
                }
            }
        }

    def create_pipeline(self, model) -> Pipeline:
        """
        Create a sklearn Pipeline with StandardScaler and the given model.

        Args:
            model: The classifier model to use.

        Returns:
            A sklearn Pipeline object.
        """
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])
        return pipeline

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities (optional).

        Returns:
            Dictionary containing calculated metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary"),
            "recall": recall_score(y_true, y_pred, average="binary"),
            "f1_score": f1_score(y_true, y_pred, average="binary"),
        }

        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def perform_cross_validation(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation and return mean scores.

        Args:
            pipeline: The model pipeline to evaluate.
            X: Feature data.
            y: Target labels.
            cv: Number of cross-validation folds.

        Returns:
            Dictionary containing mean CV scores.
        """
        logger.info(f"Performing {cv}-fold cross-validation...")

        cv_strategy = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_results = {}

        for metric in scoring_metrics:
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_strategy,
                scoring=metric,
                n_jobs=-1
            )
            cv_results[f"cv_{metric}_mean"] = scores.mean()
            cv_results[f"cv_{metric}_std"] = scores.std()
            logger.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return cv_results

    def train_model_with_tuning(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        perform_tuning: bool = True
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Train a model with hyperparameter tuning and evaluate it.

        Args:
            model_name: Name of the model to train.
            X_train: Training features.
            y_train: Training labels.
            X_test: Testing features.
            y_test: Testing labels.
            perform_tuning: Whether to perform hyperparameter tuning.

        Returns:
            Tuple containing the trained pipeline and results dictionary.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*60}")

        with mlflow.start_run(run_name=model_name):
            # Get model configuration
            configs = self.get_model_configs()
            config = configs[model_name]

            # Create pipeline
            pipeline = self.create_pipeline(config["model"])

            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("hyperparameter_tuning", perform_tuning)

            # Perform hyperparameter tuning if requested
            if perform_tuning:
                logger.info("Performing hyperparameter tuning...")
                grid_search = GridSearchCV(
                    pipeline,
                    config["param_grid"],
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)

                best_pipeline = grid_search.best_estimator_
                best_params = grid_search.best_params_

                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

                # Log best parameters
                for param_name, param_value in best_params.items():
                    mlflow.log_param(param_name, param_value)
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
            else:
                logger.info("Training with default parameters...")
                best_pipeline = pipeline
                best_pipeline.fit(X_train, y_train)
                best_params = {}

            # Perform cross-validation on best model
            cv_results = self.perform_cross_validation(best_pipeline, X_train, y_train)

            # Log CV metrics
            for metric_name, metric_value in cv_results.items():
                mlflow.log_metric(metric_name, metric_value)

            # Make predictions
            y_train_pred = best_pipeline.predict(X_train)
            y_test_pred = best_pipeline.predict(X_test)

            y_train_pred_proba = best_pipeline.predict_proba(X_train)[:, 1]
            y_test_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_train_pred, y_train_pred_proba)
            test_metrics = self.calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

            # Log training metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
                logger.info(f"Train {metric_name}: {metric_value:.4f}")

            # Log testing metrics
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
                logger.info(f"Test {metric_name}: {metric_value:.4f}")

            # Generate and log classification report
            logger.info("\nClassification Report (Test Set):")
            report = classification_report(y_test, y_test_pred)
            logger.info(f"\n{report}")

            # Log confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            logger.info(f"\nConfusion Matrix (Test Set):\n{cm}")

            # Save model artifacts
            model_path = self.output_path / f"{model_name}_model.pkl"
            pipeline_path = self.output_path / f"{model_name}_pipeline.pkl"
            scaler_path = self.output_path / "scaler.pkl"

            # Save complete pipeline
            joblib.dump(best_pipeline, pipeline_path)
            logger.info(f"Pipeline saved to: {pipeline_path}")

            # Save model only (without preprocessing)
            joblib.dump(best_pipeline.named_steps["classifier"], model_path)
            logger.info(f"Model saved to: {model_path}")

            # Save scaler separately
            joblib.dump(best_pipeline.named_steps["scaler"], scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")

            # Log model to MLflow
            mlflow.sklearn.log_model(best_pipeline, "model")
            mlflow.log_artifact(str(pipeline_path))
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(scaler_path))

            # Prepare results
            results = {
                "model_name": model_name,
                "best_params": best_params,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_results": cv_results,
                "model_path": str(model_path),
                "pipeline_path": str(pipeline_path),
            }

            return best_pipeline, results

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models_to_train: List[str] = None,
        perform_tuning: bool = True
    ) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
        """
        Train all specified models and return results.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Testing features.
            y_test: Testing labels.
            models_to_train: List of model names to train (None for all).
            perform_tuning: Whether to perform hyperparameter tuning.

        Returns:
            Dictionary mapping model names to (pipeline, results) tuples.
        """
        if models_to_train is None:
            models_to_train = list(self.get_model_configs().keys())

        logger.info(f"Training models: {models_to_train}")

        all_results = {}

        for model_name in models_to_train:
            try:
                pipeline, results = self.train_model_with_tuning(
                    model_name,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    perform_tuning
                )
                all_results[model_name] = (pipeline, results)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        # Compare models
        self._compare_models(all_results)

        return all_results

    def _compare_models(self, results: Dict[str, Tuple[Pipeline, Dict[str, Any]]]):
        """
        Compare all trained models and log summary.

        Args:
            results: Dictionary of training results.
        """
        logger.info(f"\n{'='*60}")
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info(f"{'='*60}")

        comparison_data = []

        for model_name, (_, result) in results.items():
            test_metrics = result["test_metrics"]
            comparison_data.append({
                "Model": model_name,
                "Accuracy": test_metrics["accuracy"],
                "Precision": test_metrics["precision"],
                "Recall": test_metrics["recall"],
                "F1-Score": test_metrics["f1_score"],
                "ROC-AUC": test_metrics["roc_auc"]
            })

        comparison_df = pd.DataFrame(comparison_data)
        logger.info(f"\n{comparison_df.to_string(index=False)}")

        # Find best model by ROC-AUC
        best_model = comparison_df.loc[comparison_df["ROC-AUC"].idxmax()]
        logger.info(f"\nBest Model: {best_model['Model']} (ROC-AUC: {best_model['ROC-AUC']:.4f})")

        # Save comparison to CSV
        comparison_path = self.output_path / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to: {comparison_path}")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train machine learning models for heart disease prediction"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed",
        help="Path to preprocessed data directory (default: data/processed)"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="models",
        help="Path to save trained models (default: models)"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="heart_disease_prediction",
        help="MLflow experiment name (default: heart_disease_prediction)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic_regression", "random_forest", "xgboost"],
        help="Models to train (default: all models)"
    )

    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning (use default parameters)"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def main():
    """
    Main function to orchestrate model training.
    """
    # Parse arguments
    args = parse_arguments()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize trainer
        trainer = ModelTrainer(
            data_path=args.data_path,
            output_path=args.output_path,
            experiment_name=args.experiment_name,
            random_state=args.random_state
        )

        # Load data
        X_train, y_train, X_test, y_test = trainer.load_data()

        # Train models
        results = trainer.train_all_models(
            X_train,
            y_train,
            X_test,
            y_test,
            models_to_train=args.models,
            perform_tuning=not args.no_tuning
        )

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Models saved to: {args.output_path}")
        logger.info(f"MLflow experiment: {args.experiment_name}")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
