# export_model.py
import mlflow
from mlflow import MlflowClient
import joblib
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

MODEL_REGISTRY_NAME = "BreastCancerChampion"

# Load champion model from registry using alias
champion_model = mlflow.sklearn.load_model(f"models:/{MODEL_REGISTRY_NAME}@champion")

# Export to model.pkl
joblib.dump(champion_model, "model.pkl")
print("Champion model exported to model.pkl")
