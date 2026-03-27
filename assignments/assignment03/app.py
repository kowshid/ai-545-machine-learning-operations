# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

class PredictionRequest(BaseModel):
    features: list[float]  # 30 features for breast cancer dataset

@app.get("/")
def root():
    return {"message": "Breast Cancer Inference Service"}

@app.post("/predict")
def predict(request: PredictionRequest):
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].tolist()
    return {
        "prediction": int(prediction),
        "label": "malignant" if prediction == 0 else "benign",
        "probabilities": {"malignant": probability[0], "benign": probability[1]}
    }
