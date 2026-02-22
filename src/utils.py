import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


def setup_mlflow():

    dagshub_token = os.getenv("DAGSHUB_TRACKING_PASSWORD")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TRACKING_PASSWORD is not set")

    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI is not set")

    # Set credentials first
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # Then set tracking URI
    mlflow.set_tracking_uri(tracking_uri)