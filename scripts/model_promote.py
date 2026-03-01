from mlflow.tracking import MlflowClient
from logger.logger import get_logger
from src.utils import setup_mlflow

logger = get_logger("promote_to_production")

def promote_model(model_name: str):
    try:
        setup_mlflow()
        client = MlflowClient()

        logger.info("Fetching latest staging model...")
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not staging_versions:
            raise Exception("No model found in Staging stage")

        version = staging_versions[0].version

        logger.info(f"Promoting version {version} to Production")

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

        logger.info("Model promoted to Production successfully")

    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise e


if __name__ == "__main__":
    promote_model("my_model")