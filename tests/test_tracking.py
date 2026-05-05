import os
import mlflow
import dagshub
from dotenv import load_dotenv

def test_mlflow_dagshub_connection():
    # Load environment variables from .env file
    load_dotenv()

    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=os.getenv("DAGSHUB_REPO"),
        mlflow=True
    )

    try:
        with mlflow.start_run(run_name="pytest_verification"):
            mlflow.log_param("test_type", "automated_pytest")
            mlflow.log_metric("connection_status", 1.0)
        connection_success = True
    except Exception as e:
        print(f"Connection failed: {e}")
        connection_success = False

    assert connection_success == True