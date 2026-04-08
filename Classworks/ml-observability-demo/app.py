from fastapi import FastAPI
import time
import random
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI()

# Define metrics
REQUEST_COUNT = Counter('request_count', 'Total requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')


# Default root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}


@app.get("/predict")
def predict():
    start_time = time.time()

    # Simulated prediction logic
    prediction = random.choice([0, 1])

    # Increment counter and record latency
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"prediction": prediction}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")