from src.utils import setup_mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import mlflow
import pytest


@pytest.fixture(scope="session", autouse=True)
def mlflow_setup():
    setup_mlflow()


@pytest.fixture(scope="session")
def model_name():
    return "my_model"


@pytest.fixture(scope="session")
def mlflow_client(mlflow_setup):
    return MlflowClient()


@pytest.fixture(scope="session")
def staging_model(model_name, mlflow_client, mlflow_setup):
    versions = mlflow_client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        pytest.fail('no staging model found')
    version = versions[0].version
    return mlflow.pyfunc.load_model(f'models:/{model_name}/{version}')


@pytest.fixture(scope="session")
def production_model(model_name, mlflow_client, mlflow_setup):
    versions = mlflow_client.get_latest_versions(model_name, stages=['Production'])
    if not versions:
        return None
    version = versions[0].version
    return mlflow.pyfunc.load_model(f'models:/{model_name}/{version}')


@pytest.fixture(scope="session")
def x_test_data():
    return pd.read_csv('data/proccessed/x_test.csv')


@pytest.fixture(scope="session")
def y_test_data():
    return pd.read_csv('data/proccessed/y_test.csv')