import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from logger.logger import get_logger
import pickle 
import os
import json
import mlflow
from src.utils import setup_mlflow
from mlflow.models.signature import infer_signature

logger=get_logger('Evaluation')

setup_mlflow()

def load_data(x_test_path:str,y_test_path:str)->tuple[pd.DataFrame,pd.Series]:
    try:
        logger.info(f'laoding data for testing from the {x_test_path} and {y_test_path}')
        x_test=pd.read_csv(x_test_path)
        y_test=pd.read_csv(y_test_path)
        logger.info(f'data loaded successfully from the {x_test_path} and {y_test_path}')
        return x_test,y_test
    except FileNotFoundError as e:
        logger.error(f'File not found from the {x_test_path} and {y_test_path}:{e}')
    except Exception as e:
        logger.error('an error occurred while loading data:{e}')


def load_model(model_path:str)->Pipeline:
    try:
        logger.info(f'loading the model from the {model_path}')
        model=pickle.load(open(model_path,'rb'))
        logger.info(f'model loaded successfully at:{model_path}')
        return model
    except FileNotFoundError as e:
        logger.error(f'File not found from the {model_path}:{e}')
    except Exception as e:
        logger.error(f'an unexpected error occurred while loading the model:{e}')


def model_evaluation(model:Pipeline,x_test:pd.DataFrame,y_test:pd.Series)->dict:
    try:
        logger.info('model evaluation started')
        y_pred=model.predict(x_test)    

        mse=mean_squared_error(y_test,y_pred)
        logger.info(f'mse calculated successfully {mse}')

        mae=mean_absolute_error(y_test,y_pred) 
        logger.info(f'mae calculated successfully{mae}')

        r2_scr=r2_score(y_test,y_pred)
        logger.info(f'r2_scr calculated successfully{r2_scr}')

        logger.info('model evaluation completed successfully')

        metrics={'mae':mae,'mse':mse,'r2_scr':r2_scr}
        logger.info(f'metrics dict created:{metrics}')

        return metrics
    except Exception as e:
        logger.error(f'an error occurred while evaluating the model:{e}')
        raise


def saving_evaluation_metric(metrics:dict,save_metrics_path:str)->None:
    try:
        logger.info(f'saving the evaluation metrics at the {save_metrics_path}')
        os.makedirs(save_metrics_path,exist_ok=True)

        with open(os.path.join(save_metrics_path,'metrics.json'),'w') as f:
            json.dump(metrics,f,indent=4)
        logger.info(f'metrics saved successfully at the {save_metrics_path}')    

    except Exception as e:
        logger.error(f'an error occurred while saving the metrics:{e}')   
        


def save_run_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "model_path": model_path}, f, indent=4)
        logger.info(f"Run info saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Failed to save run info:{str(e)}")
            


def main():
    mlflow.set_experiment('Dvc-Pipeline')
    with mlflow.start_run(run_name='model') as run:

        save_run_info(run.info.run_id,'model','reports/run_info.json')

        x_test,y_test=load_data('data/proccessed/x_test.csv','data/proccessed/y_test.csv')

        model=load_model('models/model.pkl')

        metrics=model_evaluation(model,x_test,y_test)
         
        # loger metrices
        mlflow.log_metrics(metrics)
        # log params and model
        mlflow.log_params(model.named_steps['model'].get_params())
        signature=infer_signature(x_test,model.predict(x_test))
        mlflow.sklearn.log_model(model,'model',input_example=x_test.head(1),signature=signature)

        saving_evaluation_metric(metrics,'reports')

        mlflow.log_artifact('reports/metrics.json')



if __name__ == "__main__":
    main()        






