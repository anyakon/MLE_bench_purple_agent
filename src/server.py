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
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Fill in your agent card
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="mle-bench-ml-solver",
        name="ML Competition Solver",
        description="Automatically solves machine learning competitions from MLE-Bench. Accepts competition.tar.gz with Kaggle-style ML tasks (tabular, classification, regression) and returns predictions in submission.csv format. Uses AutoML-inspired approaches with Gradient Boosting, Random Forests, and intelligent feature engineering.",
        tags=["machine-learning", "kaggle", "autoML", "classification", "regression", "tabular-data", "mle-bench"],
        examples=[
            """
            {
                "participants": {
                    "agent": "http://localhost:8000/"
                },
                "config": {
                    "competition_id": "spaceship-titanic"
                }
            }
            """
        ]
    )

    agent_card = AgentCard(
        name="MLE-Bench Purple Agent",
        description="Machine Learning engineering agent that automatically solves Kaggle competitions from MLE-Bench. Accepts competition data archives, analyzes task structure, trains appropriate ML models (Gradient Boosting, Random Forests), and returns predictions in standard submission.csv format. Supports binary classification, multiclass classification, and regression tasks on tabular data.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text', 'application/gzip'],
        default_output_modes=['text', 'text/csv'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        max_content_length=None
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
