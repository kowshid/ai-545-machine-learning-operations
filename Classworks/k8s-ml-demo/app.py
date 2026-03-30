import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris Inference API")
model = joblib.load("model.pkl")


class IrisRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "Iris inference service v3 is running"}


@app.post("/predict")
def predict(data: IrisRequest):
    prediction = model.predict([data.features])[0]
    return {"prediction": int(prediction)}
