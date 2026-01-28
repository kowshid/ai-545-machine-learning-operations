import pandas as pd
from pathlib import Path

RAW = Path("data/raw/customers.csv")
OUT = Path("data/processed/customers_clean.csv")
df = pd.read_csv(RAW)
df = df.dropna(subset=["age", "income"])
df = df[df["age"].between(18, 90)]
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("Wrote:", OUT)
