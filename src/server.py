"""
Purple Agent A2A Server

Single-service architecture:
1. Accepts competition.tar.gz via A2A protocol
2. Solves it with PurpleAgent (ML models + OpenAI LLM)
3. Returns submission.csv

Architecture: server.py → executor.py → agent.py → purple_agent.py
"""
import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the Purple ML Agent server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="URL to advertise in agent card")
    parser.add_argument("--openai-model", type=str, default="gpt-5.1", help="OpenAI model name")
    args = parser.parse_args()

    skill = AgentSkill(
        id="mle-bench-ml-solver",
        name="ML Competition Solver",
        description=(
            "Accepts competition.tar.gz with Kaggle-style tabular ML tasks "
            "(binary/multiclass classification, regression), analyzes the data, "
            "selects the best ML models via OpenAI LLM + cross-validation, "
            "trains an ensemble, and returns submission.csv with predictions."
        ),
        tags=[
            "machine-learning", "kaggle", "autoML",
            "classification", "regression", "tabular-data",
            "mle-bench", "openai", "ensemble",
        ],
        examples=[
            "Send competition.tar.gz as FilePart with mime_type='application/gzip'"
        ],
    )

    agent_card = AgentCard(
        name="MLE-Bench Purple Agent",
        description=(
            "Machine Learning engineering agent that solves Kaggle competitions. "
            "Accepts competition.tar.gz archive, analyzes task structure, "
            "uses OpenAI GPT for strategy recommendation, trains ML ensembles "
            "via cross-validation, and returns submission.csv with predictions. "
            "Supports binary classification, multiclass classification, and regression."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="2.0.0",
        default_input_modes=["text", "application/gzip"],
        default_output_modes=["text", "text/csv"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        max_content_length=None,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
