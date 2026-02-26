import json, mlflow
from mlflow.tracking import MlflowClient
from src.utils import setup_mlflow
from logger.logger import get_logger

logger = get_logger("model_registry")

def load_model_info(path: str) -> dict:
    try:
        logger.info(f"fetching model info from {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("model info loaded successfully")
        return data
    except Exception as e:
        logger.error(f"failed to load model info: {e}")
        raise

def register_model(model_info: dict, client: MlflowClient) -> None:
    try:
        logger.info("model registry process started")
        name, run_id, path = "my_model", model_info["run_id"], model_info["model_path"]
        uri = f"runs:/{run_id}/{path}"
        try:
            client.get_registered_model(name)
            logger.info(f"registered model '{name}' already exists")
        except:
            client.create_registered_model(name)
            logger.info(f"registered model '{name}' created")
        version = client.create_model_version(name=name, source=uri, run_id=run_id)
        logger.info(f"model version {version.version} created successfully")
        client.transition_model_version_stage(name=name, version=version.version, stage="Staging", archive_existing_versions=True)
        logger.info(f"model version {version.version} transitioned to Staging")
        logger.info("model registry process completed successfully")
    except Exception as e:
        logger.error(f"model registry failed: {e}")
        raise

def main():
    try:
        setup_mlflow()
        model_info = load_model_info("reports/run_info.json")
        register_model(model_info, MlflowClient())
    except Exception as e:
        logger.error(f"registry stage execution failed: {e}")
        raise

if __name__ == "__main__":
    main()