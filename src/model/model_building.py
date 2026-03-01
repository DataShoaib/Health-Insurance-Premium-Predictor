import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from src.features.feature_engineering import pipeline_preprocessor
from sklearn.pipeline import Pipeline
from logger.logger import get_logger
import os
import pickle
import mlflow
from mlflow.models.signature import infer_signature
from src.utils import setup_mlflow
import json

logger = get_logger('model_building')


def load_data(x_train_path: str, y_train_path: str) -> tuple[pd.DataFrame, pd.Series]:
    try:
        logger.info(f'loading the training data from {x_train_path} and {y_train_path}')
        return pd.read_csv(x_train_path), pd.read_csv(y_train_path)
    except Exception as e:
        logger.error(f'error while loading training data: {e}')
        raise


def model_training(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    try:
        logger.info('model training started')
        pipe = Pipeline([
            ('preprocessor', pipeline_preprocessor()),
            ('model', GradientBoostingRegressor(learning_rate=0.1, n_estimators=300, max_depth=3))
        ])
        pipe.fit(x_train, y_train)
        logger.info('model trained successfully')
        return pipe
    except Exception as e:
        logger.error(f'error during model training: {e}')
        raise


def save_model(model: Pipeline, save_model_path: str) -> None:
    try:
        logger.info(f'saving model at {save_model_path}')
        os.makedirs(save_model_path, exist_ok=True)
        pickle.dump(model, open(os.path.join(save_model_path, 'model.pkl'), 'wb'))
        logger.info('model saved successfully')
    except Exception as e:
        logger.error(f'error while saving model: {e}')
        raise


def save_run_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        json.dump({"run_id": run_id, "model_path": model_path}, open(file_path, "w"), indent=4)
        logger.info(f'run info saved at {file_path}')
    except Exception as e:
        logger.error(f'error saving run info: {e}')
        raise


def main():
    try:
        setup_mlflow()

        mlflow.set_experiment('insurance-Predictor-Final')

        with mlflow.start_run(run_name='training') as run:

            x_train, y_train = load_data('data/proccessed/x_train.csv', 'data/proccessed/y_train.csv')
            model = model_training(x_train, y_train)

            mlflow.log_params(model.named_steps['model'].get_params())

            signature = infer_signature(x_train, model.predict(x_train))

            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=x_train.head(1))

            save_run_info(run.info.run_id, "model", "reports/run_info.json")

            save_model(model, 'models')

            logger.info('training stage completed successfully')

    except Exception as e:
        logger.error(f'training stage failed: {e}')
        raise

if __name__ == '__main__':
    main()