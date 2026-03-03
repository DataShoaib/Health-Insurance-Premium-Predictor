import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


def setup_mlflow():

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not all([username, password, tracking_uri]):
        print("MLflow credentials missing. Skipping setup.")
        return False

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    return True