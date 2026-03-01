from fastapi.testclient import TestClient
from app.main import app
import requests

client=TestClient(app)

def test_api():
    response=client.get("/Health")
    assert response.json()== {"status":"Health-Insurance-Premium-Predictor is running"}
    assert response.status_code==200

def test_predict_endpoint():
    payload={'age':25,"gender":"male","bmi":22.1,"bloodpressure":100,"diabetic":"No","children":2,"smoker":"No","region":"southwest"}
    response=client.post("/predict",json=payload)
    assert response.status_code==200
    assert "claim" in response.json()
    assert response.json()['claim']>0
    