from typing import Any
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and required fields of an agent card."""
    errors: list[str] = []

    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

    # Verify Purple Agent specific fields
    assert card_data["name"] == "MLE-Bench Purple Agent"
    assert "application/gzip" in card_data["defaultInputModes"]
    assert "text/csv" in card_data["defaultOutputModes"]
    assert any(s["id"] == "mle-bench-ml-solver" for s in card_data["skills"])

# Add your custom tests here

# Purple Agent Tests

import base64
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from purple_agent import (
    DataPreprocessor,
    LLMDecisionMaker,
    ModelTrainer,
    PurpleAgent,
    TaskAnalyzer,
)


class TestTaskAnalyzer:
    """Tests for TaskAnalyzer class."""

    def test_analyze_binary_classification(self, tmp_path):
        """Test analysis of binary classification task."""
        train_df = pd.DataFrame({
            "id": range(100),
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.choice([0, 1], size=100),
        })

        test_df = pd.DataFrame({
            "id": range(20),
            "feature1": np.random.randn(20),
            "feature2": np.random.randn(20),
        })

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        task_info = TaskAnalyzer.analyze_competition(tmp_path)

        assert task_info["task_type"] == "binary_classification"
        assert task_info["target_column"] == "target"
        assert task_info["id_column"] == "id"
        assert task_info["target_cardinality"] == 2

    def test_analyze_multiclass_classification(self, tmp_path):
        """Test analysis of multiclass classification task."""
        train_df = pd.DataFrame({
            "id": range(100),
            "feature1": np.random.randn(100),
            "label": np.random.choice(["A", "B", "C"], size=100),
        })

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)

        task_info = TaskAnalyzer.analyze_competition(tmp_path)

        assert task_info["task_type"] == "multiclass_classification"
        assert task_info["target_column"] == "label"

    def test_analyze_regression(self, tmp_path):
        """Test analysis of regression task."""
        train_df = pd.DataFrame({
            "Id": range(100),
            "feature1": np.random.randn(100),
            "Target": np.random.randn(100) * 100 + 50,
        })

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)

        task_info = TaskAnalyzer.analyze_competition(tmp_path)

        assert task_info["task_type"] == "regression"
        assert task_info["target_column"] == "Target"
        assert task_info["target_cardinality"] == "continuous"

    def test_find_data_files(self, tmp_path):
        """Test finding data files in competition directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").touch()
        (data_dir / "test.csv").touch()
        (data_dir / "description.md").touch()

        files = TaskAnalyzer._find_data_files(tmp_path)
        assert len(files) == 3
        assert any(f.name == "train.csv" for f in files)

    def test_read_description(self, tmp_path):
        """Test reading competition description."""
        desc_file = tmp_path / "description.md"
        desc_file.write_text("# Test Competition\nThis is a test.")

        description = TaskAnalyzer._read_description(tmp_path)
        assert "Test Competition" in description

    def test_analyze_csv_stats(self, tmp_path):
        """Test that _analyze_csv returns extended stats."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "id": range(200),
            "f1": np.random.randn(200),
            "cat": np.random.choice(["A", "B", "C"], size=200),
            "target": np.random.choice([0, 1], size=200),
        })
        # Add some missing values
        train_df.loc[:19, "f1"] = np.nan

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)

        info = TaskAnalyzer._analyze_csv(data_dir / "train.csv")
        assert info["n_rows"] == 200
        assert info["n_cols"] == 4
        # cat=1 categorical, id+f1 = 2 numeric (target excluded from numeric count)
        assert info["n_categorical"] == 1
        assert info["n_numeric"] >= 1  # at least f1
        assert info["missing_ratio"] > 0


class TestLLMDecisionMaker:
    """Tests for LLMDecisionMaker class."""

    def test_heuristic_fallback_no_llm(self):
        """Test that heuristic fallback works without LLM."""
        dm = LLMDecisionMaker(llm_fn=None)
        task_info = {
            "task_type": "binary_classification",
            "n_rows": 1000,
            "n_cols": 20,
            "n_categorical": 3,
            "missing_ratio": 0.05,
        }
        rec = dm.recommend(task_info)
        assert "models" in rec
        assert "preprocessing" in rec
        assert len(rec["models"]) >= 2

    def test_llm_invocation(self):
        """Test that LLM is called when available."""
        mock_response = json.dumps({
            "task_type": "binary_classification",
            "models": ["lightgbm", "random_forest", "xgboost"],
            "preprocessing": ["scaling"],
            "feature_engineering": [],
            "justification": "test",
        })
        mock_llm = MagicMock(return_value=mock_response)

        dm = LLMDecisionMaker(llm_fn=mock_llm)
        task_info = {
            "task_type": "binary_classification",
            "n_rows": 5000,
            "n_cols": 30,
            "n_categorical": 5,
            "missing_ratio": 0.02,
            "target_cardinality": 2,
            "class_balance": {0: 2500, 1: 2500},
        }
        rec = dm.recommend(task_info)

        mock_llm.assert_called_once()
        assert rec["models"] == ["lightgbm", "random_forest", "xgboost"]

    def test_openai_client_created_with_key(self):
        """Test that OpenAI client is initialized when key is provided."""
        dm = LLMDecisionMaker(
            openai_api_key="sk-test-key-12345",
            openai_model="gpt-4.1",
        )
        # Client should be initialized if openai package is available
        assert dm._openai_client is not None
        assert dm._openai_model == "gpt-4.1"

    def test_openai_uses_env_key(self):
        """Test that OpenAI uses OPENAI_API_KEY from environment."""
        import os
        old = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-env-test-key"
        try:
            dm = LLMDecisionMaker()
            # Should pick up env key
            assert dm._openai_client is not None
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_llm_failure_fallback(self):
        """Test fallback when LLM raises an exception."""
        mock_llm = MagicMock(side_effect=RuntimeError("LLM unavailable"))

        dm = LLMDecisionMaker(llm_fn=mock_llm)
        task_info = {"task_type": "regression", "n_rows": 100, "n_cols": 10,
                      "n_categorical": 0, "missing_ratio": 0.0}
        rec = dm.recommend(task_info)

        assert "models" in rec  # fallback result

    def test_build_prompt(self):
        """Test prompt building."""
        task_info = {
            "task_type": "binary_classification",
            "target_column": "target",
            "id_column": "id",
            "n_rows": 500,
            "n_cols": 15,
            "n_numeric": 10,
            "n_categorical": 4,
            "missing_ratio": 0.03,
            "target_cardinality": 2,
            "class_balance": {0: 300, 1: 200},
            "description": "Predict survival.",
        }
        prompt = LLMDecisionMaker._build_prompt(task_info)
        assert "binary_classification" in prompt
        assert "Predict survival." in prompt

    def test_parse_llm_response_with_fences(self):
        """Test parsing LLM response with markdown fences."""
        text = '```json\n{"task_type": "regression", "models": ["ridge"], "preprocessing": [], "feature_engineering": [], "justification": "ok"}\n```'
        result = LLMDecisionMaker._parse_llm_response(text)
        assert result is not None
        assert result["task_type"] == "regression"

    def test_parse_invalid_response(self):
        """Test parsing invalid response returns None."""
        result = LLMDecisionMaker._parse_llm_response("not json at all")
        assert result is None


import json  # needed for mock responses above


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_fit_transform_binary(self):
        """Test preprocessing for binary classification."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "id": range(100),
            "num1": np.random.randn(100),
            "num2": np.random.randn(100),
            "cat1": np.random.choice(["A", "B"], size=100),
            "target": np.random.choice([0, 1], size=100),
        })
        test_df = pd.DataFrame({
            "id": range(100, 110),
            "num1": np.random.randn(10),
            "num2": np.random.randn(10),
            "cat1": np.random.choice(["A", "B"], size=10),
        })

        prep = DataPreprocessor()
        X_train, y_train, X_test = prep.fit_transform(
            train_df, test_df, target_col="target", id_col="id",
            task_type="binary_classification",
        )

        assert X_train.shape[0] == 100
        assert X_test.shape[0] == 10
        assert y_train is not None
        assert len(y_train) == 100
        assert X_train.dtype == np.float32

    def test_fit_transform_with_missing_values(self):
        """Test imputation of missing values."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "id": range(100),
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "cat": np.random.choice(["X", "Y"], size=100),
            "target": np.random.choice([0, 1], size=100),
        })
        train_df.loc[:19, "f1"] = np.nan
        train_df.loc[:9, "cat"] = np.nan

        test_df = pd.DataFrame({
            "id": range(100, 105),
            "f1": np.random.randn(5),
            "f2": np.random.randn(5),
            "cat": np.random.choice(["X"], size=5),
        })
        test_df.loc[:1, "f2"] = np.nan

        prep = DataPreprocessor()
        X_train, y_train, X_test = prep.fit_transform(
            train_df, test_df, "target", "id", "binary_classification",
        )

        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_fit_transform_regression(self):
        """Test preprocessing for regression task."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "Id": range(50),
            "feature": np.random.randn(50),
            "Target": np.random.randn(50) * 10 + 50,
        })
        test_df = train_df.drop(columns=["Target"]).iloc[:5]

        prep = DataPreprocessor()
        X_train, y_train, X_test = prep.fit_transform(
            train_df, test_df, "Target", "Id", "regression",
        )

        assert y_train is not None
        assert len(y_train) == 50

    def test_inverse_transform_target(self):
        """Test inverse transform of predictions."""
        train_df = pd.DataFrame({
            "id": range(20),
            "f": range(20),
            "label": ["yes"] * 10 + ["no"] * 10,
        })
        test_df = train_df.drop(columns=["label"]).iloc[:3]

        prep = DataPreprocessor()
        X_train, y_train, X_test = prep.fit_transform(
            train_df, test_df, "label", "id", "binary_classification",
        )

        preds = np.array([0, 1, 0])
        original = prep.inverse_transform_target(preds)
        assert list(original) == ["no", "yes", "no"]

    def test_feature_engineering(self):
        """Test that engineered features are created."""
        np.random.seed(42)
        n = 50
        train_df = pd.DataFrame({
            "id": range(n),
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
            "d": np.random.randn(n),
            "target": np.random.choice([0, 1], size=n),
        })
        test_df = train_df.drop(columns=["target"]).iloc[:5]

        prep = DataPreprocessor()
        X_train, _, X_test = prep.fit_transform(
            train_df, test_df, "target", "id", "binary_classification",
        )

        # Should have more features than original 4 due to interactions
        original_n_features = 4
        assert X_train.shape[1] > original_n_features


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_train_binary_classification(self, tmp_path):
        """Test training on binary classification task."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "PassengerId": range(100),
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "Survived": np.random.choice([0, 1], size=100),
        })

        test_df = pd.DataFrame({
            "PassengerId": range(100, 120),
            "feature1": np.random.randn(20),
            "feature2": np.random.randn(20),
        })

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        task_info = {
            "task_type": "binary_classification",
            "target_column": "Survived",
            "id_column": "PassengerId",
            "strategy": "gradient_boosting",
        }

        submission_df = ModelTrainer.train_and_predict(tmp_path, task_info)

        assert len(submission_df) == 20
        assert "PassengerId" in submission_df.columns
        assert "Survived" in submission_df.columns

    def test_train_with_categoricals(self, tmp_path):
        """Test training with categorical features."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "id": range(100),
            "category": np.random.choice(["A", "B", "C"], size=100),
            "value": np.random.randn(100),
            "target": np.random.choice([0, 1], size=100),
        })

        test_df = pd.DataFrame({
            "id": range(100, 110),
            "category": np.random.choice(["A", "B"], size=10),
            "value": np.random.randn(10),
        })

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        task_info = {
            "task_type": "binary_classification",
            "target_column": "target",
            "id_column": "id",
        }

        submission_df = ModelTrainer.train_and_predict(tmp_path, task_info)

        assert len(submission_df) == 10
        assert not submission_df["target"].isna().any()

    def test_train_with_missing_values(self, tmp_path):
        """Test training with missing values in data."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "id": range(100),
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
        })
        train_df.loc[:9, "feature1"] = np.nan
        train_df["target"] = np.random.choice([0, 1], size=100)

        test_df = pd.DataFrame({
            "id": range(100, 115),
            "feature1": np.random.randn(15),
            "feature2": np.random.randn(15),
        })
        test_df.loc[:4, "feature2"] = np.nan

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        task_info = {
            "task_type": "binary_classification",
            "target_column": "target",
            "id_column": "id",
        }

        submission_df = ModelTrainer.train_and_predict(tmp_path, task_info)

        assert len(submission_df) == 15
        assert not submission_df["target"].isna().any()

    def test_train_regression(self, tmp_path):
        """Test training on regression task."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "Id": range(80),
            "f1": np.random.randn(80),
            "f2": np.random.randn(80),
            "Target": np.random.randn(80) * 10 + 50,
        })
        test_df = pd.DataFrame({
            "Id": range(80, 95),
            "f1": np.random.randn(15),
            "f2": np.random.randn(15),
        })

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        task_info = {
            "task_type": "regression",
            "target_column": "Target",
            "id_column": "Id",
        }

        submission_df = ModelTrainer.train_and_predict(tmp_path, task_info)

        assert len(submission_df) == 15
        assert "Id" in submission_df.columns
        assert "Target" in submission_df.columns

    def test_select_best_model(self, tmp_path):
        """Test model selection via cross-validation."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "id": range(200),
            "f1": np.random.randn(200),
            "f2": np.random.randn(200),
            "f3": np.random.randn(200),
            "target": np.random.choice([0, 1], size=200),
        })
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        (data_dir / "test.csv").write_text(
            pd.DataFrame({"id": [0], "f1": [0], "f2": [0], "f3": [0]}).to_csv(index=False)
        )

        task_info = {
            "task_type": "binary_classification",
            "target_column": "target",
            "id_column": "id",
        }

        submission_df = ModelTrainer.train_and_predict(
            tmp_path, task_info, use_ensemble=False, n_folds=3,
        )
        assert len(submission_df) == 1
        assert not submission_df["target"].isna().any()

    def test_ensemble_training(self, tmp_path):
        """Test that ensemble mode produces valid predictions."""
        np.random.seed(42)
        train_df = pd.DataFrame({
            "id": range(100),
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "target": np.random.choice([0, 1], size=100),
        })
        test_df = pd.DataFrame({
            "id": range(100, 110),
            "f1": np.random.randn(10),
            "f2": np.random.randn(10),
        })

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        task_info = {
            "task_type": "binary_classification",
            "target_column": "target",
            "id_column": "id",
        }

        submission_df = ModelTrainer.train_and_predict(
            tmp_path, task_info, use_ensemble=True, n_folds=3,
        )
        assert len(submission_df) == 10


class TestPurpleAgent:
    """Tests for PurpleAgent class."""

    def test_extract_competition_data(self):
        """Test extraction of competition.tar.gz."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            with tarfile.open(fileobj=f, mode="w:gz") as tar:
                data_dir = tempfile.mkdtemp()
                (Path(data_dir) / "train.csv").touch()
                (Path(data_dir) / "test.csv").touch()
                tar.add(data_dir, arcname="home/data")
                import shutil
                shutil.rmtree(data_dir)
            f.seek(0)
            tar_bytes = f.read()

        agent = PurpleAgent()
        try:
            competition_dir = agent.extract_competition_data(tar_bytes)
            assert competition_dir.exists()
        finally:
            agent.cleanup()

    def test_create_submission_bytes(self):
        """Test creating submission CSV bytes."""
        agent = PurpleAgent()

        submission_df = pd.DataFrame({
            "id": [1, 2, 3],
            "target": [0.1, 0.5, 0.9],
        })

        csv_bytes = agent.create_submission_bytes(submission_df)

        result_df = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(result_df) == 3
        assert "id" in result_df.columns

    def test_solve_competition_pipeline(self):
        """Test full competition solving pipeline."""
        np.random.seed(42)

        train_df = pd.DataFrame({
            "id": range(200),
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "feature3": np.random.choice(["A", "B", "C"], size=200),
            "target": np.random.choice([0, 1], size=200),
        })

        test_df = pd.DataFrame({
            "id": range(200, 250),
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "feature3": np.random.choice(["A", "B"], size=50),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "home" / "data"
            data_dir.mkdir(parents=True)

            train_df.to_csv(data_dir / "train.csv", index=False)
            test_df.to_csv(data_dir / "test.csv", index=False)
            (data_dir.parent / "description.md").write_text("# Test Competition")

            tar_path = Path(tmpdir) / "competition.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(Path(tmpdir) / "home", arcname="home")

            tar_bytes = tar_path.read_bytes()

            agent = PurpleAgent()
            try:
                submission_bytes = agent.solve_competition(tar_bytes)

                submission_df = pd.read_csv(io.BytesIO(submission_bytes))
                assert len(submission_df) == 50
                assert "target" in submission_df.columns
            finally:
                agent.cleanup()

    def test_cleanup(self):
        """Test that cleanup removes temporary files."""
        agent = PurpleAgent()

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            with tarfile.open(fileobj=f, mode="w:gz") as tar:
                temp_dir = tempfile.mkdtemp()
                tar.add(temp_dir, arcname="home/data")
                import shutil
                shutil.rmtree(temp_dir)
            f.seek(0)
            tar_bytes = f.read()

        agent.extract_competition_data(tar_bytes)
        work_dir = agent.work_dir

        assert work_dir is not None
        assert work_dir.exists()

        agent.cleanup()
        assert not work_dir.exists()
        assert agent.work_dir is None

    def test_with_llm_fn(self):
        """Test PurpleAgent with custom LLM function."""
        def dummy_llm(prompt: str) -> str:
            return json.dumps({
                "task_type": "binary_classification",
                "models": ["random_forest", "gradient_boosting"],
                "preprocessing": ["scaling"],
                "feature_engineering": [],
                "justification": "dummy",
            })

        agent = PurpleAgent(llm_fn=dummy_llm)
        assert agent.llm_fn is dummy_llm

    def test_with_openai_params(self):
        """Test PurpleAgent with OpenAI parameters."""
        agent = PurpleAgent(
            openai_api_key="sk-test-123",
            openai_model="gpt-4.1",
        )
        assert agent.openai_api_key == "sk-test-123"
        assert agent.openai_model == "gpt-4.1"
