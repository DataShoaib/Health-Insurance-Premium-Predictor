import dagshub
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

def setup_mlflow():
    mlflow.set_tracking_uri()
    dagshub_token=os.getenv('DAGSHUB_TRACKING_PASSWORD')
    if not dagshub_token :
        raise EnvironmentError('DAGSHUB_TRACKING_PASSWORD IS NOT SET')
    
    os.environ['MLFLOW_TRACKING_USERNAME']= dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD']= dagshub_token

    mlflow.set_tracking_uri(os.getenv['MLFLOW_TRACKING_URI'])

