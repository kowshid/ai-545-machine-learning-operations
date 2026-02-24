import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC  # Changed from RandomForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mlflow.set_experiment("breast_cancer_experiment")


def main():
    # 1. Load Data
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Note: SVMs are sensitive to scaling, so we scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. MLflow Run
    with mlflow.start_run() as run:
        # Initialize SVC instead of Random Forest
        model = SVC(kernel='linear', C=1.0, random_state=42)

        # Fit the model
        model.fit(X_train_scaled, y_train)

        # Predict and Score
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)

        print(f"Accuracy: {acc:.4f}")

        # Log Parameters, Metrics, and Model
        mlflow.log_param("kernel", "linear")
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", acc)

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="BreastCancerClassifier"
        )

        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()