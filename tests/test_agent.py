from typing import Any
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
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

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
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

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)

# Add your custom tests here

# Purple Agent Tests

import base64
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from purple_agent import ModelTrainer, PurpleAgent, TaskAnalyzer


class TestTaskAnalyzer:
    """Tests for TaskAnalyzer class."""

    def test_analyze_binary_classification(self, tmp_path):
        """Test analysis of binary classification task."""
        # Create mock data
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

        # Create directory structure
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        # Analyze
        task_info = TaskAnalyzer.analyze_competition(tmp_path)

        assert task_info["task_type"] == "binary_classification"
        assert task_info["target_column"] == "target"
        assert task_info["id_column"] == "id"
        assert "gradient_boosting" in task_info["strategy"]

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
            "Target": np.random.randn(100) * 100 + 50,  # Continuous values
        })

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)

        task_info = TaskAnalyzer.analyze_competition(tmp_path)

        assert task_info["task_type"] == "regression"
        assert task_info["target_column"] == "Target"

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


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_train_binary_classification(self, tmp_path):
        """Test training on binary classification task."""
        # Create train/test data
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

        # Train and predict
        submission_df = ModelTrainer.train_and_predict(tmp_path, task_info)

        assert len(submission_df) == 20
        assert "PassengerId" in submission_df.columns
        assert "target" in submission_df.columns
        # Predictions should be probabilities (between 0 and 1)
        assert submission_df["target"].min() >= 0
        assert submission_df["target"].max() <= 1

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
        # Add missing values
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


class TestPurpleAgent:
    """Tests for PurpleAgent class."""

    def test_extract_competition_data(self):
        """Test extraction of competition.tar.gz."""
        # Create a mock tar.gz file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            with tarfile.open(fileobj=f, mode="w:gz") as tar:
                # Add some files
                data_dir = tempfile.mkdtemp()
                (Path(data_dir) / "train.csv").touch()
                (Path(data_dir) / "test.csv").touch()
                tar.add(data_dir, arcname="home/data")

                # Cleanup temp
                import shutil
                shutil.rmtree(data_dir)

            f.seek(0)
            tar_bytes = f.read()

        # Test extraction
        agent = PurpleAgent()
        try:
            competition_dir = agent.extract_competition_data(tar_bytes)
            assert competition_dir.exists()
            # Should have extracted files
            assert any(competition_dir.iterdir()) or any((competition_dir.parent).iterdir())
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

        # Verify it's valid CSV
        result_df = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(result_df) == 3
        assert "id" in result_df.columns

    def test_solve_competition_pipeline(self):
        """Test full competition solving pipeline."""
        np.random.seed(42)

        # Create mock competition data
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

        # Create tar.gz
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "home" / "data"
            data_dir.mkdir(parents=True)

            train_df.to_csv(data_dir / "train.csv", index=False)
            test_df.to_csv(data_dir / "test.csv", index=False)
            (data_dir.parent / "description.md").write_text("# Test Competition")

            # Create tar.gz
            tar_path = Path(tmpdir) / "competition.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(Path(tmpdir) / "home", arcname="home")

            tar_bytes = tar_path.read_bytes()

            # Solve
            agent = PurpleAgent()
            try:
                submission_bytes = agent.solve_competition(tar_bytes)

                # Verify submission
                submission_df = pd.read_csv(io.BytesIO(submission_bytes))
                assert len(submission_df) == 50
                assert "target" in submission_df.columns
            finally:
                agent.cleanup()

    def test_cleanup(self):
        """Test that cleanup removes temporary files."""
        agent = PurpleAgent()

        # Create minimal tar.gz
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
