import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.features.feature_engineering import feature_Construction
import pickle


def load_model(model_path:str)->Pipeline:
    try:
        with open(model_path,'rb') as f:
           model= pickle.load(f)
           return model
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Model File not found')
    
def predict(model:Pipeline,data:dict)->dict:
    try:
        final_data=feature_Construction(data)
        prediction=model.predict(final_data)
        return prediction
    except Exception as e:
        raise e
    
if __name__=="__main__" :
    predict()  

        
