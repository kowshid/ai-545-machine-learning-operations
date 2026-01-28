import pandas as pd
import yaml
from pathlib import Path

RAW = Path("data/raw/customers.csv")
OUT = Path("data/processed/customers_clean.csv")
PARAMS = Path("params.yml")
params = yaml.safe_load(PARAMS.read_text())

min_age = params["cleaning"]["min_age"]
max_age = params["cleaning"]["max_age"]

df = pd.read_csv(RAW)
df = df.dropna(subset=["age", "income"])
df = df[df["age"].between(min_age, max_age)]

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote: {OUT}")
