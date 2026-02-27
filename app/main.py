from fastapi import FastAPI
from app.schema import Inputdata,response
from app.config import Settings
from app.model_app_predict import load_model,predict

settings=Settings()

app=FastAPI(title=settings.app_name,version=settings.version)


model=load_model(settings.model_path)

@app.get('/Health')
def Health_check():
    return {'status':'Health-Insurance-Premium-Predictor is runnig '}

import pandas as pd

@app.post("/predict")
def model_prediction(data: Inputdata):

    input_dict = data.model_dump()

    # Convert to 1 row dataframe
    df = pd.DataFrame([input_dict])

    response = predict(model, df)

    return {"prediction": response.tolist()}
