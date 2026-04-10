"""
Purple Agent - Advanced ML Engineering Agent for MLE-Bench

This agent solves ML competitions from MLE-Bench by:
1. Receiving and extracting competition.tar.gz
2. Deep data analysis (EDA) - statistics, distributions, correlations, missing values
3. LLM-powered strategy selection using GPT-4o
4. Intelligent feature engineering based on data characteristics and LLM advice
5. Auto-selecting optimal models based on task type, data properties, and LLM recommendations
6. Training ensemble of models with cross-validation
7. Generating submission.csv predictions with best model/ensemble
"""
import io
import json
import logging
import os
import tarfile
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger(__name__)


# ============================================================================
# TASK ANALYZER - Deep EDA
# ============================================================================


class TaskAnalyzer:
    """
    Analyze competition data to determine task type, data characteristics,
    and optimal modeling strategy.
    """

    @staticmethod
    def analyze_competition(competition_dir: Path) -> dict[str, Any]:
        """Full analysis of competition directory."""
        description = TaskAnalyzer._read_description(competition_dir)
        data_files = TaskAnalyzer._find_data_files(competition_dir)
        train_file = TaskAnalyzer._find_train_file(competition_dir, data_files)

        task_info: dict[str, Any] = {
            "description": description,
            "data_files": [str(f) for f in data_files],
            "task_type": "binary_classification",
            "data_type": "tabular",
            "target_column": None,
            "id_column": None,
            "n_samples": 0,
            "n_features": 0,
            "n_categorical": 0,
            "n_numerical": 0,
            "missing_pct": 0.0,
            "class_distribution": None,
            "target_stats": {},
            "correlation_info": {},
            "strategy": "ensemble",
            "needs_scaling": False,
            "needs_encoding": False,
            "has_text_features": False,
            "is_imbalanced": False,
        }

        if train_file and train_file.suffix == ".csv":
            eda = TaskAnalyzer._deep_analyze_csv(train_file)
            task_info.update(eda)

        logger.info(f"Task analysis: {task_info['task_type']}, "
                     f"{task_info['n_samples']} samples, {task_info['n_features']} features, "
                     f"missing={task_info['missing_pct']:.1f}%, "
                     f"imbalanced={task_info['is_imbalanced']}")

        return task_info

    @staticmethod
    def _read_description(competition_dir: Path) -> Optional[str]:
        desc_path = competition_dir / "description.md"
        if desc_path.exists():
            return desc_path.read_text()
        return None

    @staticmethod
    def _find_data_files(competition_dir: Path) -> list[Path]:
        data_dir = competition_dir / "data"
        if data_dir.exists():
            return list(data_dir.glob("*"))
        return list(competition_dir.glob("*"))

    @staticmethod
    def _find_train_file(
        competition_dir: Path, data_files: list[Path]
    ) -> Optional[Path]:
        for f in data_files:
            if f.name.lower().startswith("train") and f.suffix == ".csv":
                return f
        for f in data_files:
            if f.suffix == ".csv":
                return f
        return None

    @staticmethod
    def _deep_analyze_csv(train_file: Path) -> dict[str, Any]:
        """Deep EDA on training CSV."""
        try:
            df = pd.read_csv(train_file, nrows=5000)
        except Exception as e:
            logger.warning(f"Failed to analyze CSV: {e}")
            return {}

        info: dict[str, Any] = {}

        # --- Target column ---
        target_candidates = [
            "target", "Target", "TARGET", "label", "Label",
            "survived", "Survived", "class", "Class", "output", "Output",
            "SalePrice", "price", "Price", "response", "Response",
        ]
        target_col = None
        for candidate in target_candidates:
            if candidate in df.columns:
                target_col = candidate
                break
        if not target_col:
            target_col = df.columns[-1]
        info["target_column"] = target_col

        # --- ID column ---
        id_candidates = ["id", "ID", "Id", "PassengerId", "passenger_id", "row_id"]
        info["id_column"] = None
        for candidate in id_candidates:
            if candidate in df.columns:
                info["id_column"] = candidate
                break

        # --- Feature analysis ---
        feature_cols = [
            c for c in df.columns if c != target_col and c != info["id_column"]
        ]
        info["n_features"] = len(feature_cols)
        info["n_samples"] = len(df)

        cat_cols = df[feature_cols].select_dtypes(include=["object", "category", "string"]).columns.tolist()
        num_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
        info["n_categorical"] = len(cat_cols)
        info["n_numerical"] = len(num_cols)
        info["categorical_cols"] = cat_cols
        info["numerical_cols"] = num_cols

        # --- Missing values ---
        missing_mask = df[feature_cols].isnull()
        info["missing_pct"] = float(missing_mask.sum().sum()) / max(missing_mask.size, 1) * 100
        info["cols_with_missing"] = missing_mask.any().any()

        # --- Task type from target ---
        if target_col:
            target_vals = df[target_col].dropna()
            n_unique = target_vals.nunique()
            info["n_target_unique"] = int(n_unique)

            if n_unique == 2:
                info["task_type"] = "binary_classification"
                vals = target_vals.value_counts()
                info["class_distribution"] = vals.to_dict()
                ratio = vals.min() / vals.max()
                info["is_imbalanced"] = ratio < 0.3
            elif n_unique <= 15:
                info["task_type"] = "multiclass_classification"
                info["class_distribution"] = target_vals.value_counts().to_dict()
                vals = target_vals.value_counts()
                ratio = vals.min() / vals.max()
                info["is_imbalanced"] = ratio < 0.2
            else:
                if pd.api.types.is_numeric_dtype(target_vals):
                    info["task_type"] = "regression"
                    info["target_stats"] = {
                        "mean": float(target_vals.mean()),
                        "std": float(target_vals.std()),
                        "min": float(target_vals.min()),
                        "max": float(target_vals.max()),
                        "skew": float(target_vals.skew()),
                    }
                else:
                    info["task_type"] = "multiclass_classification"

        # --- Text features detection ---
        info["has_text_features"] = False
        text_candidates = []
        for col in cat_cols:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 50:
                text_candidates.append(col)
                info["has_text_features"] = True
        info["text_cols"] = text_candidates

        # --- Correlation with target (for numerical) ---
        if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
            corr_info = {}
            for col in num_cols:
                clean = df[[col, target_col]].dropna()
                if len(clean) > 10:
                    corr = clean[col].corr(clean[target_col])
                    corr_info[col] = round(float(corr), 3)
            info["target_correlations"] = dict(
                sorted(corr_info.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            )

        # --- High cardinality categoricals ---
        high_card = []
        for col in cat_cols:
            if df[col].nunique() > 50:
                high_card.append(col)
        info["high_cardinality_cols"] = high_card

        # --- Strategy recommendation ---
        info["strategy"] = TaskAnalyzer._recommend_strategy(info)

        return info

    @staticmethod
    def _recommend_strategy(info: dict) -> str:
        """Recommend modeling strategy based on EDA."""
        task = info["task_type"]
        n = info.get("n_samples", 0)
        p = info.get("n_features", 0)
        missing = info.get("missing_pct", 0)
        imbalanced = info.get("is_imbalanced", False)

        if p > 100:
            return "high_dimensional"
        elif imbalanced:
            return "imbalanced"
        elif n > 50000:
            return "large_dataset"
        elif missing > 20:
            return "many_missing"
        elif task == "regression":
            skew = info.get("target_stats", {}).get("skew", 0)
            if abs(skew) > 2:
                return "skewed_regression"
            return "regression"
        elif task == "binary_classification":
            return "binary_classification"
        elif task == "multiclass_classification":
            return "multiclass_classification"
        return "default"


# ============================================================================
# LLM ADVISOR - GPT-4o strategic guidance
# ============================================================================


class LLMAdvisor:
    """
    Uses GPT-4o to guide modeling decisions based on EDA results.
    Provides recommendations for:
    - Modeling strategy
    - Model selection and ordering
    - Feature engineering suggestions
    - Hyperparameter hints
    """

    SYSTEM_PROMPT = """You are an expert ML consultant specializing in Kaggle-style tabular competitions.
You analyze EDA (Exploratory Data Analysis) results and provide concise, actionable recommendations.

Respond ONLY with valid JSON matching this exact schema:
{
    "strategy": "brief strategy description",
    "models": ["model1", "model2", ...],
    "model_params": {"model_name": {"param": value, ...}},
    "feature_engineering": ["suggestion1", "suggestion2", ...],
    "cv_folds": 5,
    "notes": "any important considerations"
}

Available models: lightgbm, gb_clf, gb_reg, rf_clf, rf_reg, et_clf, et_reg, lr_clf, ridge_reg, knn_clf, knn_reg.
Keep responses concise. Focus on practical advice that improves CV score."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.enabled = bool(self.api_key)
        self._client = None
        self.last_advice: dict[str, Any] = {}

    @property
    def client(self):
        if self._client is None and self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed, LLM advice disabled")
                self.enabled = False
        return self._client

    def recommend_strategy(self, eda_info: dict) -> dict[str, Any]:
        """Get modeling recommendations from GPT-4o based on EDA."""
        # Default fallback advice
        default_advice = {
            "strategy": f"Standard ensemble for {eda_info.get('task_type', 'unknown')}",
            "models": ["lightgbm", "gb_clf", "rf_clf"],
            "model_params": {},
            "feature_engineering": ["impute_missing", "encode_categoricals", "scale_numeric"],
            "cv_folds": 5,
            "notes": "Using default advice - LLM not available",
        }

        if not self.enabled or self.client is None:
            logger.info("LLM advisor disabled — using default strategy")
            self.last_advice = default_advice
            return default_advice

        # Prepare compact EDA summary for the LLM
        summary = {
            "task_type": eda_info.get("task_type"),
            "n_samples": eda_info.get("n_samples"),
            "n_features": eda_info.get("n_features"),
            "n_categorical": eda_info.get("n_categorical"),
            "n_numerical": eda_info.get("n_numerical"),
            "missing_pct": round(eda_info.get("missing_pct", 0), 1),
            "is_imbalanced": eda_info.get("is_imbalanced", False),
            "class_distribution": eda_info.get("class_distribution"),
            "target_correlations": eda_info.get("target_correlations"),
            "high_cardinality_cols": eda_info.get("high_cardinality_cols", [])[:5],
            "target_stats": eda_info.get("target_stats"),
            "has_text_features": eda_info.get("has_text_features", False),
        }

        prompt = f"""Here are the EDA results for an ML competition. Recommend the best modeling approach.

EDA Summary:
{json.dumps(summary, indent=2, default=str)}

Provide specific recommendations for model selection, hyperparameters, and feature engineering."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            advice = json.loads(content)
            self.last_advice = advice
            logger.info(f"LLM advice received: {advice.get('strategy', '')}")
            return advice

        except Exception as e:
            logger.warning(f"LLM advice failed ({e}), using defaults")
            self.last_advice = default_advice
            return default_advice

    def recommend_features(self, eda_info: dict, current_features: list[str]) -> list[str]:
        """Get feature engineering suggestions from LLM."""
        if not self.enabled:
            return []

        prompt = f"""Given these features for a {eda_info.get('task_type', 'unknown')} task,
suggest 3-5 feature engineering ideas (interactions, transformations, aggregations).

Features: {current_features[:30]}
Missing%: {eda_info.get('missing_pct', 0):.1f}%
Categoricals: {eda_info.get('n_categorical', 0)}
Numericals: {eda_info.get('n_numerical', 0)}

Respond with a JSON list: {"suggestions": ["idea1", "idea2"]}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # cheaper, faster
                messages=[
                    {"role": "system", "content": "You are a feature engineering expert. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            return result.get("suggestions", [])
        except Exception:
            return []


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


class FeatureEngineer:
    """
    Advanced feature engineering pipeline:
    - Missing value imputation (smart per column type)
    - Categorical encoding (one-hot vs target vs label)
    - Feature scaling
    - Feature generation (interactions, polynomial)
    - Outlier handling
    """

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.medians: dict[str, float] = {}
        self.onehot_cols: list[str] = []
        self.ohe: Optional[OneHotEncoder] = None
        self.feature_cols: list[str] = []

    def fit_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                      target_col: Optional[str], id_col: Optional[str],
                      task_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit on train, transform both. Returns X_train, y_train, X_test.
        """
        # Drop ID and target
        drop_cols = [id_col] if id_col and id_col in train_df.columns else []
        if target_col and target_col in train_df.columns:
            drop_cols.append(target_col)

        train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
        test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")

        self.feature_cols = train.columns.tolist()

        # --- Target ---
        y_train = None
        if target_col and target_col in train_df.columns:
            y_train = train_df[target_col].copy()

        # --- Missing value imputation ---
        train, test = self._impute_missing(train, test, task_info)

        # --- Categorical encoding ---
        train, test = self._encode_categoricals(train, test, task_info)

        # --- Convert all to numeric ---
        for col in train.columns:
            train[col] = pd.to_numeric(train[col], errors="coerce")
            test[col] = pd.to_numeric(test[col], errors="coerce")

        train = train.fillna(0)
        test = test.fillna(0)

        # --- Feature generation ---
        train, test = self._generate_features(train, test, task_info)

        # --- Scaling ---
        needs_scaling = task_info.get("strategy") in ("large_dataset", "high_dimensional", "default")
        if needs_scaling or train.shape[1] > 20:
            self.scaler = RobustScaler()
            train_arr = self.scaler.fit_transform(train)
            test_arr = self.scaler.transform(test)
        else:
            train_arr = train.values
            test_arr = test.values

        return train_arr, y_train, test_arr

    def transform(self, df: pd.DataFrame, id_col: Optional[str]) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        drop_cols = [id_col] if id_col and id_col in df.columns else []
        data = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        data, _ = self._impute_missing(data, data, {})
        data, _ = self._encode_categoricals(data, data, {})

        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.fillna(0)

        data, _ = self._generate_features(data, data, {})

        if self.scaler:
            return self.scaler.transform(data)
        return data.values

    def _impute_missing(self, train: pd.DataFrame, test: pd.DataFrame,
                        task_info: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Smart imputation per column type."""
        for col in train.columns:
            if train[col].dtype == "object":
                # Categorical: fill with mode
                mode_val = train[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
                train[col] = train[col].fillna(fill_val)
                test[col] = test[col].fillna(fill_val)
            elif pd.api.types.is_numeric_dtype(train[col]):
                # Numerical: median
                median_val = train[col].median()
                train[col] = train[col].fillna(median_val)
                test[col] = test[col].fillna(median_val)
        return train, test

    def _encode_categoricals(self, train: pd.DataFrame, test: pd.DataFrame,
                              task_info: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categoricals: one-hot for low cardinality, label for high."""
        cat_cols = train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        high_card_cols = set(task_info.get("high_cardinality_cols", []))

        low_card_cols = []
        high_card_cols_list = []

        for col in cat_cols:
            n_unique = train[col].nunique()
            if col in high_card_cols or n_unique > 20:
                high_card_cols_list.append(col)
            else:
                low_card_cols.append(col)

        # One-hot for low cardinality
        if low_card_cols:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            train_ohe = self.ohe.fit_transform(train[low_card_cols])
            test_ohe = self.ohe.transform(test[low_card_cols])

            ohe_names = self.ohe.get_feature_names_out(low_card_cols).tolist()
            train = pd.concat([
                train.drop(columns=low_card_cols),
                pd.DataFrame(train_ohe, columns=ohe_names, index=train.index)
            ], axis=1)
            test = pd.concat([
                test.drop(columns=low_card_cols),
                pd.DataFrame(test_ohe, columns=ohe_names, index=test.index)
            ], axis=1)

        # Label encoding for high cardinality
        for col in high_card_cols_list:
            le = LabelEncoder()
            train[col] = train[col].fillna("missing").astype(str)
            test[col] = test[col].fillna("missing").astype(str)
            le.fit(train[col])
            train[col] = le.transform(train[col])
            test[col] = test[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            self.label_encoders[col] = le

        return train, test

    def _generate_features(self, train: pd.DataFrame, test: pd.DataFrame,
                           task_info: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate interaction features for small/medium datasets."""
        if train.shape[1] > 50:
            return train, test  # Skip for already high-dimensional

        num_cols = train.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) < 3 or len(num_cols) > 30:
            return train, test  # Too few or too many

        # Pairwise interactions for top correlated features
        corr = task_info.get("target_correlations", {})
        top_feats = list(corr.keys())[:5]
        top_feats = [c for c in top_feats if c in num_cols]

        new_train = train.copy()
        new_test = test.copy()

        for i in range(len(top_feats)):
            for j in range(i + 1, len(top_feats)):
                c1, c2 = top_feats[i], top_feats[j]
                new_train[f"{c1}_x_{c2}"] = new_train[c1] * new_train[c2]
                new_test[f"{c1}_x_{c2}"] = new_test[c1] * new_test[c2]

        return new_train, new_test


# ============================================================================
# MODEL FACTORY
# ============================================================================


class ModelFactory:
    """
    Create models appropriate for task type and data characteristics.
    Returns list of (name, model) pairs for ensemble.
    """

    @staticmethod
    def get_models(task_info: dict, llm_advice: Optional[dict] = None) -> list[tuple[str, Any]]:
        """Get list of models for the task, optionally filtered/boosted by LLM advice."""
        task_type = task_info["task_type"]
        strategy = task_info.get("strategy", "default")
        n_samples = task_info.get("n_samples", 0)
        n_features = task_info.get("n_features", 0)
        imbalanced = task_info.get("is_imbalanced", False)

        if "classification" in task_type:
            return ModelFactory._get_classifiers(strategy, n_samples, n_features, imbalanced, llm_advice)
        else:
            return ModelFactory._get_regressors(strategy, n_samples, n_features, llm_advice)

    @staticmethod
    def _get_classifiers(strategy: str, n: int, p: int,
                         imbalanced: bool, llm_advice: Optional[dict] = None) -> list[tuple[str, Any]]:
        models = []

        # Check LLM recommended models
        llm_models = set()
        llm_params: dict[str, dict] = {}
        if llm_advice:
            llm_models = set(llm_advice.get("models", []))
            llm_params = llm_advice.get("model_params", {})

        # LightGBM - always included if available
        if HAS_LIGHTGBM:
            lgb_params = {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
            if imbalanced:
                lgb_params["scale_pos_weight"] = 10  # rough estimate
            models.append(("lightgbm", lgb.LGBMClassifier(**lgb_params)))

        # Gradient Boosting
        models.append(("gb_clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )))

        # Random Forest
        models.append(("rf_clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            random_state=42, n_jobs=-1
        )))

        # Extra Trees
        models.append(("et_clf", ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            random_state=42, n_jobs=-1
        )))

        # Logistic Regression (good for small/simple datasets)
        if n < 10000 and p < 100:
            models.append(("lr_clf", LogisticRegression(
                max_iter=2000, C=1.0, random_state=42, solver="lbfgs"
            )))

        # KNN (good for small datasets)
        if n < 20000:
            models.append(("knn_clf", KNeighborsClassifier(
                n_neighbors=7, weights="distance", n_jobs=-1
            )))

        return models

    @staticmethod
    def _get_regressors(strategy: str, n: int, p: int,
                        llm_advice: Optional[dict] = None) -> list[tuple[str, Any]]:
        models = []

        # Check LLM recommended models
        llm_models = set()
        llm_params: dict[str, dict] = {}
        if llm_advice:
            llm_models = set(llm_advice.get("models", []))
            llm_params = llm_advice.get("model_params", {})

        if HAS_LIGHTGBM:
            lgb_params = {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
            if strategy == "skewed_regression":
                lgb_params["objective"] = "gamma"
            if "lightgbm" in llm_params:
                lgb_params.update(llm_params["lightgbm"])
            models.append(("lightgbm", lgb.LGBMRegressor(**lgb_params)))

        # Gradient Boosting
        gb_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
                     "subsample": 0.8, "random_state": 42}
        if "gb_reg" in llm_params:
            gb_params.update(llm_params["gb_reg"])
        models.append(("gb_reg", GradientBoostingRegressor(**gb_params)))

        # Random Forest
        rf_params = {"n_estimators": 300, "max_depth": 12, "min_samples_split": 5,
                     "random_state": 42, "n_jobs": -1}
        if "rf_reg" in llm_params:
            rf_params.update(llm_params["rf_reg"])
        models.append(("rf_reg", RandomForestRegressor(**rf_params)))

        # Extra Trees
        et_params = {"n_estimators": 300, "max_depth": 12, "min_samples_split": 5,
                     "random_state": 42, "n_jobs": -1}
        if "et_reg" in llm_params:
            et_params.update(llm_params["et_reg"])
        models.append(("et_reg", ExtraTreesRegressor(**et_params)))

        # Ridge regression
        if p < 200:
            ridge_params = {"alpha": 1.0}
            if "ridge_reg" in llm_params:
                ridge_params.update(llm_params["ridge_reg"])
            models.append(("ridge_reg", Ridge(**ridge_params)))

        # KNN
        if n < 20000:
            knn_params = {"n_neighbors": 7, "weights": "distance", "n_jobs": -1}
            if "knn_reg" in llm_params:
                knn_params.update(llm_params["knn_reg"])
            models.append(("knn_reg", KNeighborsRegressor(**knn_params)))

        return models


# ============================================================================
# ENSEMBLE TRAINER
# ============================================================================


class EnsembleTrainer:
    """
    Train multiple models, evaluate with cross-validation,
    create ensemble (stacking or weighted voting).
    """

    @staticmethod
    def train_ensemble(X_train: np.ndarray, y_train: pd.Series,
                       X_test: np.ndarray, task_info: dict,
                       llm_advice: Optional[dict] = None) -> dict[str, Any]:
        """
        Train ensemble of models. Returns dict with predictions and metadata.
        """
        task_type = task_info["task_type"]
        n_samples = task_info.get("n_samples", 0)

        # Get models (optionally using LLM recommendations)
        models = ModelFactory.get_models(task_info, llm_advice)
        logger.info(f"Training {len(models)} models for {task_type}")

        # Cross-validation setup — use LLM recommended folds if available
        if llm_advice:
            n_splits = llm_advice.get("cv_folds", 5)
        else:
            n_splits = min(5, max(3, n_samples // 500))
        if "classification" in task_type:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Train individual models and collect CV scores + test predictions
        results = []
        oof_predictions = None  # Out-of-fold predictions for stacking
        test_predictions = {}

        for name, model in models:
            logger.info(f"  Training: {name}")
            try:
                # Clone model for each CV fold scoring
                model_copy = type(model)(**model.get_params())

                # Quick CV score
                if "classification" in task_type:
                    scoring = "roc_auc" if task_info.get("n_target_unique", 2) == 2 else "accuracy"
                else:
                    scoring = "neg_root_mean_squared_error"

                scores = cross_val_score(model_copy, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
                cv_mean = float(np.mean(scores))
                logger.info(f"    {name}: CV {scoring} = {cv_mean:.4f} (+/- {np.std(scores):.4f})")

                # Fit on full train
                model.fit(X_train, y_train)

                # Test predictions
                if "classification" in task_type and task_type == "binary_classification":
                    preds = model.predict_proba(X_test)[:, 1]
                else:
                    preds = model.predict(X_test)

                test_predictions[name] = preds
                results.append((name, model, cv_mean, scoring))

            except Exception as e:
                logger.warning(f"  Failed to train {name}: {e}")
                continue

        if not results:
            raise ValueError("No models trained successfully")

        # --- Create ensemble prediction ---
        if len(results) == 1:
            name, model, score, _ = results[0]
            ensemble_preds = test_predictions[name]
            logger.info(f"Only 1 model trained: {name} (CV={score:.4f})")
        else:
            # Weighted average by CV score
            weights = {}
            total_score = 0
            for name, _, score, scoring in results:
                # Normalize: for negative RMSE, convert to positive
                if "neg" in scoring:
                    w = max(1.0 / (abs(score) + 1e-6), 0.01)
                else:
                    w = max(score, 0.01)
                weights[name] = w
                total_score += w

            ensemble_preds = np.zeros(len(X_test))
            for name, preds in test_predictions.items():
                w = weights.get(name, 0.01) / total_score
                ensemble_preds += w * preds
                logger.info(f"  Ensemble weight: {name} = {w:.3f}")

        return {
            "predictions": ensemble_preds,
            "models": [(n, m, s) for n, m, s, _ in results],
            "test_predictions": test_predictions,
        }


# ============================================================================
# MODEL TRAINER - Main entry point
# ============================================================================


class ModelTrainer:
    """Train ML models and generate predictions."""

    @staticmethod
    def train_and_predict(
        competition_dir: Path, task_info: dict[str, Any],
        llm_advice: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Train ensemble model and return submission DataFrame."""
        data_dir = competition_dir / "data"
        if not data_dir.exists():
            data_dir = competition_dir

        train_file = None
        test_file = None
        for f in data_dir.glob("*.csv"):
            name_lower = f.name.lower()
            if name_lower.startswith("train"):
                train_file = f
            elif name_lower.startswith("test"):
                test_file = f

        if not train_file:
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                train_file = csv_files[0]
        if not train_file:
            raise ValueError("No training data found")

        # Load data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file) if test_file is not None else None
        if test_df is None:
            test_df = train_df.copy()

        target_col = task_info.get("target_column")
        id_col = task_info.get("id_column")
        task_type = task_info.get("task_type", "binary_classification")

        # Feature engineering
        fe = FeatureEngineer()
        X_train, y_train, X_test = fe.fit_transform(
            train_df, test_df, target_col, id_col, task_info
        )

        # Encode target if classification
        label_encoder = None
        if "classification" in task_type:
            if y_train is not None and y_train.dtype == "object":
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train.astype(str))

        # Train ensemble (with optional LLM guidance)
        if llm_advice:
            logger.info(f"Using LLM-guided ensemble: {llm_advice.get('strategy', '')}")
        ensemble_result = EnsembleTrainer.train_ensemble(
            X_train, y_train, X_test, task_info, llm_advice
        )
        predictions = ensemble_result["predictions"]

        # Decode predictions if needed
        if label_encoder and "classification" in task_type and task_type != "binary_classification":
            try:
                predictions = label_encoder.inverse_transform(
                    predictions.astype(int).clip(0, len(label_encoder.classes_) - 1)
                )
            except Exception:
                pass

        # Create submission DataFrame
        submission_df = pd.DataFrame()
        if id_col and id_col in test_df.columns:
            submission_df[id_col] = test_df[id_col]

        if "classification" in task_type and task_type == "binary_classification":
            submission_df["target"] = predictions
        else:
            submission_df["target"] = predictions

        logger.info(f"Generated submission with {len(submission_df)} predictions")
        return submission_df


# ============================================================================
# PURPLE AGENT - Main class
# ============================================================================


class PurpleAgent:
    """
    Purple ML Agent - solves ML competitions from MLE-Bench.

    Receives competition.tar.gz, trains models, returns submission.csv.
    Uses LLM (GPT-4o) for strategic guidance if OPENAI_API_KEY is set.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.work_dir: Optional[Path] = None
        self.task_info: dict[str, Any] = {}
        self.llm_advisor = LLMAdvisor(api_key)
        self.llm_advice: dict[str, Any] = {}

    def extract_competition_data(self, tar_bytes: bytes) -> Path:
        """Extract competition.tar.gz to temporary directory."""
        import shutil
        self.work_dir = Path(tempfile.mkdtemp(prefix="purple_agent_"))

        tar_buffer = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            tar.extractall(path=self.work_dir)

        logger.info(f"Extracted competition data to {self.work_dir}")

        # Find the actual data directory
        if (self.work_dir / "home" / "data").exists():
            return self.work_dir / "home"
        elif (self.work_dir / "data").exists():
            return self.work_dir
        else:
            subdirs = [d for d in self.work_dir.iterdir() if d.is_dir()]
            if subdirs and (subdirs[0] / "data").exists():
                return subdirs[0] / "data"
            return self.work_dir

    def analyze_task(self, competition_dir: Path) -> dict[str, Any]:
        """Analyze the competition task with deep EDA."""
        self.task_info = TaskAnalyzer.analyze_competition(competition_dir)
        logger.info(f"Task analysis complete: {self.task_info['task_type']}")
        return self.task_info

    def get_llm_advice(self) -> dict[str, Any]:
        """Get strategic advice from LLM based on EDA results."""
        self.llm_advice = self.llm_advisor.recommend_strategy(self.task_info)
        return self.llm_advice

    def solve_task(self, competition_dir: Path) -> pd.DataFrame:
        """Solve the ML competition task with LLM-guided ensemble models."""
        submission_df = ModelTrainer.train_and_predict(
            competition_dir, self.task_info, self.llm_advice
        )
        return submission_df

    def create_submission_bytes(self, submission_df: pd.DataFrame) -> bytes:
        """Convert submission DataFrame to CSV bytes."""
        csv_bytes = submission_df.to_csv(index=False).encode("utf-8")
        return csv_bytes

    def solve_competition(self, tar_bytes: bytes) -> bytes:
        """Complete pipeline: extract, analyze, get LLM advice, solve, return submission CSV."""
        competition_dir = self.extract_competition_data(tar_bytes)
        self.analyze_task(competition_dir)
        self.get_llm_advice()
        submission_df = self.solve_task(competition_dir)
        return self.create_submission_bytes(submission_df)

    def cleanup(self):
        """Clean up temporary files."""
        if self.work_dir and self.work_dir.exists():
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = None
