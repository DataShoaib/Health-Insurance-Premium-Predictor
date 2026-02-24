import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from src.features.feature_engineering import pipeline_preprocessor
from sklearn.pipeline import Pipeline
from logger.logger import get_logger
import os
import pickle

logger=get_logger('model_building')



def load_data(x_train_path:str,y_train_path:str)->tuple[pd.DataFrame,pd.Series]:
    try:
        logger.info(f'loading the training data from the {x_train_path} and {y_train_path}')
        x_train=pd.read_csv(x_train_path)
        y_train=pd.read_csv(y_train_path)
        logger.info(f'training datav loaded successfully from the {x_train_path} and {y_train_path}')
        return x_train,y_train
    except FileNotFoundError as e:
        logger.error(f'File not Found at:{x_train_path} and {y_train_path}:{e}')
    except Exception as e:
        logger.error(f'an unexpected error accured while file loading {e}')


def model_training(x_train:pd.DataFrame,y_train:pd.Series)->Pipeline:        
    try:
        logger.info('model training started')
        preproccessor=pipeline_preprocessor()
        model_pipeline=Pipeline(steps=[('preprocessor',preproccessor),('model',GradientBoostingRegressor(learning_rate=0.1,n_estimators=300,max_depth=3))])
        model_pipeline.fit(x_train,y_train)
        logger.info('model trained successfully')
        return model_pipeline
    except Exception as e:
        logger.error(f'an unexpected error accured while model training:{e}')


def save_model(model:Pipeline,save_model_path:str)->None:
    try:
        logger.info(f'saving the model at:{save_model_path}')      
        os.makedirs(save_model_path,exist_ok=True)
        with open(os.path.join(save_model_path,'model.pkl'),'wb') as f:
            pickle.dump(model,f)
        logger.info(f'model saved successfully at:{save_model_path}')
    except Exception as e:
        logger.error(f'an unexpected error accured while model.pkl saving:{e}')


def main():
   
    x_train,y_train=load_data('data/proccessed/x_train.csv','data/proccessed/y_train.csv')

    model_pipe=model_training(x_train,y_train)

    save_model(model_pipe,'models')

if __name__=='__main__':
    main()