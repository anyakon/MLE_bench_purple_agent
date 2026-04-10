"""
Purple Agent - A2A Agent that solves ML competitions.

Accepts competition.tar.gz, trains models, returns submission.csv.
"""
import base64
import logging
from pathlib import Path
from typing import Any, Optional

from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import get_message_text, new_agent_text_message

from purple_agent import PurpleAgent as MLAgent

logger = logging.getLogger(__name__)


class Agent:
    """
    A2A agent that receives competition.tar.gz and returns submission.csv.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model or "gpt-4.1"
        self._ml_agent: Optional[MLAgent] = None

    def _extract_tar_bytes(self, msg: Message) -> bytes:
        """Extract competition.tar.gz bytes from message parts."""
        for part in msg.parts:
            if isinstance(part.root, FilePart):
                file_data = part.root.file
                if isinstance(file_data, FileWithBytes):
                    return base64.b64decode(file_data.bytes)

        raise ValueError(
            "No file attachment found in message. "
            "Please send competition.tar.gz as FilePart."
        )

    def _extract_instructions(self, msg: Message) -> str:
        """Extract text instructions from message parts."""
        texts = []
        for part in msg.parts:
            if isinstance(part.root, TextPart):
                texts.append(part.root.text)
        return "\n".join(texts) if texts else ""

    async def run(self, msg: Message, updater: TaskUpdater) -> None:
        """
        Main execution pipeline:
        1. Extract competition.tar.gz
        2. Solve with ML agent
        3. Return submission.csv as artifact
        """
        # Step 1: Extract archive
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Extracting competition data...")
        )

        try:
            tar_bytes = self._extract_tar_bytes(msg)
        except ValueError as e:
            await updater.failed(new_agent_text_message(str(e)))
            return

        instructions = self._extract_instructions(msg)
        logger.info(f"Received competition data: {len(tar_bytes)} bytes")

        # Step 2: Solve
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Analyzing task, selecting models, and training ensemble..."
            )
        )

        try:
            self._ml_agent = MLAgent(
                openai_api_key=self.openai_api_key,
                openai_model=self.openai_model,
            )
            submission_bytes = self._ml_agent.solve_competition(tar_bytes)
            self._ml_agent.cleanup()
        except Exception as e:
            logger.exception("ML solving failed")
            if self._ml_agent:
                self._ml_agent.cleanup()
            await updater.failed(new_agent_text_message(f"ML solving failed: {e}"))
            return

        # Step 3: Return artifact
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Returning submission.csv...")
        )

        await updater.add_artifact(
            parts=[
                Part(root=FilePart(
                    file=FileWithBytes(
                        bytes=base64.b64encode(submission_bytes).decode("ascii"),
                        name="submission.csv",
                        mime_type="text/csv",
                    )
                ))
            ],
            name="submission",
        )
