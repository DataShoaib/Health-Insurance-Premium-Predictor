import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


def setup_mlflow():

    dagshub_token = os.getenv("DAGSHUB_TRACKING_PASSWORD")
    tracking_uri = 'https://dagshub.com/DataShoaib/Health-Insurance-Premium-predictor.mlflow'

    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TRACKING_PASSWORD is not set")


    # set credentials first
    os.environ["MLFLOW_TRACKING_USERNAME"] = "DataShoaib"
    os.environ["DAGSHUB_TRACKING_PASSWORD"] = dagshub_token

    # set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

   
    mlflow.set_registry_uri(tracking_uri)