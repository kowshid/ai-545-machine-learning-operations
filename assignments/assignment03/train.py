# train.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = "breast_cancer_classification"
mlflow.set_experiment(EXPERIMENT_NAME)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ("LogisticRegression_C1", LogisticRegression(C=1.0, max_iter=10000)),
    ("LogisticRegression_C01", LogisticRegression(C=0.1, max_iter=10000)),
    ("RandomForest_100", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("RandomForest_200", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
]

for run_name, model in models:
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        recall = recall_score(y_test, preds)
        accuracy = accuracy_score(y_test, preds)

        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"{run_name} → Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
