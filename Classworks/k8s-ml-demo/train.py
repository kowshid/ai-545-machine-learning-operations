import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y = load_iris(return_X_y=True)
# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
# Save model
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
