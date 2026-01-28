import pandas as pd
from pathlib import Path

DATA = Path("data/processed/customers_clean.csv")
OUT = Path("reports/validation.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

lines = []
lines.append(f"rows={len(df)}")
lines.append(f"cols={len(df.columns)}")
lines.append(f"age_min={df['age'].min()}")
lines.append(f"age_max={df['age'].max()}")
lines.append(f"income_min={df['income'].min()}")
lines.append(f"income_max={df['income'].max()}")

OUT.write_text("\n".join(lines) + "\n")
print(f"Wrote: {OUT}")
