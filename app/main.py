# Imports for Prometheus metrics and request tracking
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from fastapi import FastAPI, Request
import time

# Existing project imports
from app.schema import Inputdata, response
from app.config import Settings
from app.model_app_predict import load_model, predict
import pandas as pd

# App setup
settings = Settings()
app = FastAPI(title=settings.app_name, version=settings.version)
model = load_model(settings.model_path)

# Metrics

# total number of HTTP requests
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

# time taken for each request
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

# total number of predictions made
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions served"
)

# stores the last prediction value
LAST_CLAIM_AMOUNT = Gauge(
    "last_predicted_claim_amount",
    "Last predicted insurance claim value"
)

# endpoint for Prometheus to collect metrics
app.mount("/metrics", make_asgi_app())

# middleware to track all incoming requests
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.time()

    response = await call_next(request)

    duration = time.time() - start

    # update request count
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()

    # record how long the request took
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

    return response

# basic health check
@app.get('/')
def Health_check():
    return {"status": "Health-Insurance-Premium-Predictor is running"}

# prediction endpoint
@app.post("/predict", response_model=response)
def model_prediction(data: Inputdata):
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])
    pred = float(predict(model, df)[0])

    # update prediction metrics
    PREDICTION_COUNT.inc()
    LAST_CLAIM_AMOUNT.set(pred)

    return {"claim": pred}