"""
Purple Agent - ML Engineering Agent for MLE-Bench

This agent solves ML competitions from MLE-Bench by:
1. Receiving and extracting competition.tar.gz
2. Analyzing the task type (classification, regression, etc.)
3. Preprocessing data appropriately
4. Training ML models
5. Generating submission.csv predictions
"""
import base64
import io
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class TaskAnalyzer:
    """Analyze competition data to determine task type and strategy."""

    @staticmethod
    def analyze_competition(competition_dir: Path) -> dict[str, Any]:
        """
        Analyze competition directory to determine:
        - Task type (classification, regression, etc.)
        - Data type (tabular, text, image)
        - Target column(s)
        - Appropriate modeling strategy
        """
        description = TaskAnalyzer._read_description(competition_dir)
        data_files = TaskAnalyzer._find_data_files(competition_dir)
        train_file = TaskAnalyzer._find_train_file(competition_dir, data_files)

        task_info = {
            "description": description,
            "data_files": data_files,
            "task_type": "binary_classification",  # default
            "data_type": "tabular",  # default
            "target_column": None,
            "id_column": None,
            "strategy": "random_forest",
        }

        if train_file and train_file.suffix == ".csv":
            task_info.update(TaskAnalyzer._analyze_csv(train_file))

        return task_info

    @staticmethod
    def _read_description(competition_dir: Path) -> Optional[str]:
        """Read competition description if available."""
        desc_path = competition_dir / "description.md"
        if desc_path.exists():
            return desc_path.read_text()
        return None

    @staticmethod
    def _find_data_files(competition_dir: Path) -> list[Path]:
        """Find all data files in competition directory."""
        data_dir = competition_dir / "data"
        if data_dir.exists():
            return list(data_dir.glob("*"))
        return list(competition_dir.glob("*"))

    @staticmethod
    def _find_train_file(
        competition_dir: Path, data_files: list[Path]
    ) -> Optional[Path]:
        """Find training data file."""
        # Look for train.csv, train*.csv, or first CSV file
        for f in data_files:
            if f.name.lower().startswith("train") and f.suffix == ".csv":
                return f
        for f in data_files:
            if f.suffix == ".csv":
                return f
        return None

    @staticmethod
    def _analyze_csv(train_file: Path) -> dict[str, Any]:
        """Analyze CSV file to determine task characteristics."""
        try:
            df = pd.read_csv(train_file, nrows=1000)
        except Exception as e:
            logger.warning(f"Failed to analyze CSV: {e}")
            return {}

        info = {}

        # Infer target column (usually last column or specified in common names)
        target_candidates = [
            "target",
            "Target",
            "TARGET",
            "label",
            "Label",
            "survived",
            "Survived",
            "class",
            "Class",
            "output",
            "Output",
        ]

        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break

        if not target_col:
            # Assume last column is target
            target_col = df.columns[-1]

        info["target_column"] = target_col

        # Find ID column
        id_candidates = [
            "id",
            "ID",
            "Id",
            "PassengerId",
            "passenger_id",
        ]
        for candidate in id_candidates:
            if candidate in df.columns:
                info["id_column"] = candidate
                break

        # Determine task type from target
        if target_col:
            target_values = df[target_col].dropna()
            n_unique = target_values.nunique()

            if n_unique == 2:
                info["task_type"] = "binary_classification"
                info["strategy"] = "gradient_boosting"
            elif n_unique <= 10:
                info["task_type"] = "multiclass_classification"
                info["strategy"] = "random_forest"
            else:
                # Check if continuous
                if pd.api.types.is_numeric_dtype(target_values):
                    info["task_type"] = "regression"
                    info["strategy"] = "gradient_boosting_regressor"
                else:
                    info["task_type"] = "multiclass_classification"
                    info["strategy"] = "random_forest"

        return info


class ModelTrainer:
    """Train ML models and generate predictions."""

    @staticmethod
    def train_and_predict(
        competition_dir: Path, task_info: dict[str, Any]
    ) -> pd.DataFrame:
        """
        Train model and return predictions as DataFrame.
        Returns DataFrame with predictions ready for submission.
        """
        data_dir = competition_dir / "data"
        if not data_dir.exists():
            data_dir = competition_dir

        train_file = None
        test_file = None

        # Find train and test files
        for f in data_dir.glob("*.csv"):
            name_lower = f.name.lower()
            if name_lower.startswith("train"):
                train_file = f
            elif name_lower.startswith("test"):
                test_file = f

        if not train_file:
            # Use first CSV as train
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                train_file = csv_files[0]

        if not train_file:
            raise ValueError("No training data found")

        # Load data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file) if test_file is not None else None

        if test_df is None:
            # If no test file, create predictions from train
            test_df = train_df.copy()

        target_col = task_info.get("target_column")
        id_col = task_info.get("id_column")
        task_type = task_info.get("task_type", "binary_classification")

        # Prepare features
        feature_cols = [
            col
            for col in train_df.columns
            if col != target_col and col != id_col
        ]

        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col] if target_col else None
        X_test = test_df[feature_cols].copy()

        # Handle missing values
        X_train = X_train.fillna(X_train.median(numeric_only=True))
        X_test = X_test.fillna(X_train.median(numeric_only=True))

        # Encode categorical columns
        X_train, X_test = ModelTrainer._encode_categoricals(X_train, X_test)

        # Convert to numeric
        X_train = X_train.apply(pd.to_numeric, errors="coerce")
        X_test = X_test.apply(pd.to_numeric, errors="coerce")

        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Train model based on task type
        if y_train is not None:
            # Encode target if classification
            label_encoder = None
            if "classification" in task_type:
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)

            model = ModelTrainer._get_model(task_type)
            model.fit(X_train, y_train)

            # Generate predictions
            if "classification" in task_type:
                if task_type == "binary_classification":
                    predictions = model.predict_proba(X_test)[:, 1]
                else:
                    predictions = model.predict(X_test)
                    if label_encoder:
                        predictions = label_encoder.inverse_transform(
                            predictions.astype(int)
                        )
            else:
                predictions = model.predict(X_test)
        else:
            # No target - return zeros
            predictions = np.zeros(len(X_test))

        # Create submission DataFrame
        submission_df = pd.DataFrame()
        if id_col and id_col in test_df.columns:
            submission_df[id_col] = test_df[id_col]

        # Add predictions
        if "classification" in task_type and task_type == "binary_classification":
            submission_df["target"] = predictions
        else:
            submission_df["target"] = predictions

        return submission_df

    @staticmethod
    def _encode_categoricals(
        train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical columns using label encoding."""
        for col in train_df.columns:
            if train_df[col].dtype == "object":
                # Fit on train, transform both
                le = LabelEncoder()
                train_df[col] = train_df[col].fillna("missing")
                test_df[col] = test_df[col].fillna("missing")

                # Handle unseen categories in test
                train_vals = train_df[col].astype(str)
                test_vals = test_df[col].astype(str)

                le.fit(train_vals)
                train_df[col] = le.transform(train_vals)

                # Handle unknown categories
                test_df[col] = test_vals.apply(
                    lambda x: le.transform([str(x)])[0]
                    if str(x) in le.classes_
                    else -1
                )

        return train_df, test_df

    @staticmethod
    def _get_model(task_type: str):
        """Get appropriate model for task type."""
        if task_type == "binary_classification":
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        elif task_type == "multiclass_classification":
            return RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif task_type == "regression":
            return GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            )
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)


class PurpleAgent:
    """
    Purple ML Agent - solves ML competitions from MLE-Bench.

    Receives competition.tar.gz, trains models, returns submission.csv.
    """

    def __init__(self):
        self.work_dir: Optional[Path] = None
        self.task_info: dict[str, Any] = {}

    def extract_competition_data(self, tar_bytes: bytes) -> Path:
        """
        Extract competition.tar.gz to temporary directory.
        Returns path to extracted directory.
        """
        self.work_dir = Path(tempfile.mkdtemp(prefix="purple_agent_"))

        tar_buffer = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            tar.extractall(path=self.work_dir)

        logger.info(f"Extracted competition data to {self.work_dir}")

        # Find the actual data directory
        # MLE-Bench typically extracts to home/data structure
        if (self.work_dir / "home" / "data").exists():
            return self.work_dir / "home"
        elif (self.work_dir / "data").exists():
            return self.work_dir
        else:
            # Check if there's a subdirectory
            subdirs = [d for d in self.work_dir.iterdir() if d.is_dir()]
            if subdirs and (subdirs[0] / "data").exists():
                return subdirs[0] / "data"
            return self.work_dir

    def analyze_task(self, competition_dir: Path) -> dict[str, Any]:
        """Analyze the competition task."""
        self.task_info = TaskAnalyzer.analyze_competition(competition_dir)
        logger.info(f"Task analysis complete: {self.task_info}")
        return self.task_info

    def solve_task(self, competition_dir: Path) -> pd.DataFrame:
        """
        Solve the ML competition task.
        Returns submission DataFrame.
        """
        submission_df = ModelTrainer.train_and_predict(
            competition_dir, self.task_info
        )
        logger.info(
            f"Generated submission with {len(submission_df)} predictions"
        )
        return submission_df

    def create_submission_bytes(self, submission_df: pd.DataFrame) -> bytes:
        """
        Convert submission DataFrame to CSV bytes.
        """
        csv_bytes = submission_df.to_csv(index=False).encode("utf-8")
        return csv_bytes

    def solve_competition(self, tar_bytes: bytes) -> bytes:
        """
        Complete pipeline: extract, analyze, solve, return submission CSV.
        Returns submission.csv as bytes.
        """
        # Extract
        competition_dir = self.extract_competition_data(tar_bytes)

        # Analyze
        self.analyze_task(competition_dir)

        # Solve
        submission_df = self.solve_task(competition_dir)

        # Return CSV bytes
        return self.create_submission_bytes(submission_df)

    def cleanup(self):
        """Clean up temporary files."""
        if self.work_dir and self.work_dir.exists():
            import shutil

            shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = None
