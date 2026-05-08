import os
import mlflow
import dagshub
from dotenv import load_dotenv
from src import paths

def test_mlflow_dagshub_connection() -> None:
    # Load environment variables from .env file

    # Load environment variables from .env file in the project root
    env_path = paths.PROJECT_ROOT / ".env"
    load_dotenv(env_path)

    # Guard clauses
    repo_owner = os.getenv("DAGSHUB_USERNAME")
    repo_name = os.getenv("DAGSHUB_REPO")

    assert repo_owner is not None, "DAGSHUB_USERNAME is not set in environment variables."
    assert repo_name is not None, "DAGSHUB_REPO is not set in environment variables."

    # Initialize DagsHub connection
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=os.getenv("DAGSHUB_REPO"),
        mlflow=True
    )

    mlflow.set_experiment("Integration_Test_Experiment")

    with mlflow.start_run(run_name="pytest_verification"):
        mlflow.log_param("test_type", "automated_pytest")
        mlflow.log_metric("connection_status", 1.0)