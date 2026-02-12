# cleans and preprocesses the raw data. I want only take out the following
# flower rows: Setosas for which Petal width is less than 0.2, Versicolor for which Sepal width
# is more than 3 and Virginica for which sepal length is more than 6.5. Make sure these rules
# are not magic numbers in your code.
# "sepal.length","sepal.width","petal.length","petal.width","variety"


from pathlib import Path

import pandas as pd
import yaml

RAW = Path("data/raw/iris.csv")
OUT = Path("data/processed/iris_cleaned.csv")
PARAMS = Path("params.yml")
params = yaml.safe_load(PARAMS.read_text())

max_petal_width = params["cleaning"]["max_petal_width"]
min_sepal_width = params["cleaning"]["min_sepal_width"]
min_sepal_length = params["cleaning"]["min_sepal_length"]

df = pd.read_csv(RAW)

target_setosa = (df["variety"] == "Setosa") & (df["petal.width"] < max_petal_width)
target_versicolor = (df["variety"] == "Versicolor") & (df["sepal.width"] > min_sepal_width)
target_virginica = (df["variety"] == "Virginica") & (df["sepal.length"] > min_sepal_length)

final_df = df[target_setosa | target_versicolor | target_virginica]

OUT.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUT, index=False)
print(f"Wrote: {OUT}")
