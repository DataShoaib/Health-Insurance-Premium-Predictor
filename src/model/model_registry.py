import logging
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient
from src.utils import setup_mlflow
from logger.logger import get_logger
import json

logger=get_logger('model_registry')


def load_model_info(model_info_path:str)->dict:
    try:
        logger.info(f'fetching the model_info from the  {model_info_path}')
        with open(model_info_path,'r') as f:
            model_info=json.load(f)
        logger.info(f'model_info loaded successfully from the {model_info_path}')   
        return model_info
    except FileNotFoundError as e:
        logger.error(f'File not found error {e}') 
    except Exception as e:
        logger.error(f'an error occurred while loading the model_info:{e}')

def model_registry(model_info:dict,client:MlflowClient)->None:
    try:
        logger.info('model registry started')
        model_path=model_info['model_path']
        run_id=model_info['run_id']        
        model_uri=f'runs:/{run_id}/{model_path}'

        model_name='my_model'

        model_version=mlflow.register_model(model_uri=model_uri,model_name=model_name)

        client.transition_model_version_stage(name=model_name,version=model_version.version,stage='Staging')

        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error(f'Error during model registration: {e}')

def main():
    setup_mlflow()
    model_info=load_model_info(model_info_path='reports/run_info.json')

    client=MlflowClient()
    model_registry(model_info=model_info,client=client)

if __name__=='__main__':
    main()    



