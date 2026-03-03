import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from logger.logger import get_logger
import pickle
import os
import json
import mlflow
from src.utils import setup_mlflow

logger = get_logger('Evaluation')

def load_data(x_test_path: str, y_test_path: str):
    try:
        logger.info(f'loading test data from {x_test_path} and {y_test_path}')
        return pd.read_csv(x_test_path), pd.read_csv(y_test_path)
    except Exception as e:
        logger.error(f'error loading test data: {e}')
        raise


def load_model(model_path: str) -> Pipeline:
    try:
        logger.info(f'loading model from {model_path}')
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f'error loading model: {e}')
        raise


def model_evaluation(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        logger.info('evaluation started')
        y_pred = model.predict(x_test)

        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2_scr': r2_score(y_test, y_pred)
        }

        logger.info(f'evaluation completed with metrics {metrics}')
        return metrics

    except Exception as e:
        logger.error(f'error during evaluation: {e}')
        raise


def save_metrics(metrics: dict, save_path: str) -> None:
    try:
        logger.info(f'saving metrics to {save_path}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        logger.info('metrics.json saved successfully')
    except Exception as e:
        logger.error(f'error saving metrics.json: {e}')
        raise


def main():
    try:
        setup_mlflow()
        run_info = json.load(open('reports/run_info.json'))
        run_id = run_info['run_id']

        x_test, y_test = load_data(
            'data/proccessed/x_test.csv',
            'data/proccessed/y_test.csv'
        )

        model = load_model('models/model.pkl')

        metrics = model_evaluation(model, x_test, y_test)

        # Save locally for DVC
        save_metrics(metrics, 'reports/metrics.json')

        # Log to existing MLflow run
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            logger.info('metrics logged to existing MLflow run')

        logger.info('evaluation stage completed successfully')

    except Exception as e:
        logger.error(f'evaluation stage failed: {e}')
        raise


if __name__ == "__main__":
    main()