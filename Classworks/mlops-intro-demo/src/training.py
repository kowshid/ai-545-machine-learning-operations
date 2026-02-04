import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load parameters from YAML file
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Churn_Prediction_Iterations")

# Load simple dataset
df = pd.read_csv("data/data.csv")
X = df[["age", "income"]]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["data"]["test_size"],
    random_state=params["model"]["random_state"]
)

# Loop through the array of max_iter values
for iteration_value in params["model"]["max_iters"]:
    with mlflow.start_run(run_name=f"max_iter_{iteration_value}"):
        # Initialize model with the current iteration value
        model = LogisticRegression(max_iter=iteration_value)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log parameters
        mlflow.log_param("max_iter", iteration_value)
        mlflow.log_param("test_size", params["data"]["test_size"])
        mlflow.log_param("random_state", params["model"]["random_state"])

        # Log metrics and model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Finished run with max_iter={iteration_value}, Accuracy: {acc}")