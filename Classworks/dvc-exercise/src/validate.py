import pandas as pd
import json
from pathlib import Path

# Input/Output paths
INPUT_FILE = Path("data/processed/iris_cleaned.csv")
OUTPUT_FILE = Path("reports/metrics.json")


df = pd.read_csv(INPUT_FILE)

feature_cols = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

metrics = {}

grouped = df.groupby("variety")

for name, group in grouped:
    metrics[name] = {}

    for col in feature_cols:
        # Calculate basic stats
        stats = group[col].describe()

        # Calculate specific metrics
        mean_val = stats['mean']
        median_val = group[col].median()
        std_val = stats['std']
        min_val = stats['min']
        max_val = stats['max']
        range_val = max_val - min_val

        # Nest metrics under the column name
        metrics[name][col] = {
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "range": round(range_val, 4)
        }

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics written to {OUTPUT_FILE}")
