import warnings

import joblib
import mlflow
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI()
# Load model.pkl at startup
print("Loading model from model.pkl...")
model = joblib.load("model.pkl")
print("Model loaded:", type(model))


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    if len(input_data.features) != 30:
        return {"error": "Expected 30 features"}
    arr = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(arr)
    return {
        "prediction": int(prediction[0])
    }
