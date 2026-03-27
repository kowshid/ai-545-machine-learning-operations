# register_champion.py
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

EXPERIMENT_NAME = "breast_cancer_classification"
MODEL_REGISTRY_NAME = "BreastCancerChampion"

# Find experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.recall DESC"],
    max_results=1
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_recall = best_run.data.metrics["recall"]
print(f"Best Run ID: {best_run_id}, Recall: {best_recall:.4f}")

# Register the best model
model_uri = f"runs:/{best_run_id}/model"
registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

# Assign "champion" alias to this version
client.set_registered_model_alias(
    name=MODEL_REGISTRY_NAME,
    alias="champion",
    version=registered_model.version
)
print(f"Registered version {registered_model.version} as 'champion'")
