import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = pd.read_csv("data/data.csv")
X = df[["age", "income"]]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["data"]["test_size"],
    random_state=params["model"]["random_state"]
)

with mlflow.start_run():
    model = LogisticRegression(max_iter=params["model"]["max_iter"])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_params(params["model"])
    mlflow.log_param("test_size", params["data"]["test_size"])

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

print(f"Accuracy: {acc}")