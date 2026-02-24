import joblib
import mlflow.sklearn

MODEL_NAME = "BreastCancerSVC" # Updated to match previous step
MODEL_ALIAS = "champion"
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print(f"--- Fetching Model: {model_uri} ---")
sk_model = mlflow.sklearn.load_model(model_uri)

# Print Model Details
print(f"Model Type: {type(sk_model)}")
print(f"Hyperparameters: {sk_model.get_params()}")

# Specific details for SVC
if hasattr(sk_model, "support_vectors_"):
    print(f"Number of Support Vectors: {len(sk_model.support_vectors_)}")
    print(f"Classes: {sk_model.classes_}")
    print(f"Dual Coefficients shape: {sk_model.dual_coef_.shape}")

# Save as plain pickle
joblib.dump(sk_model, "model.pkl")
print("--- Saved model.pkl successfully ---")