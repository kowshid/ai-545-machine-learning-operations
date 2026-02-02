import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Load simple dataset
df = pd.read_csv("data.csv")
X = df[["age", "income"]]
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
with mlflow.start_run():
    model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
mlflow.log_param("model_type", "logistic_regression")
mlflow.log_param("max_iter", 500)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(
    model,
    "model",
)
print("Accuracy:", acc)
