from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/customers.csv")
PROCESSED_PATH = Path("data/processed/customers_clean.csv")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # simple cleaning: drop NA, filter invalid ages
    df = df.dropna(subset=["age"])
    df = df[df["age"].between(18, 90)]
    return df


def validate_data(df: pd.DataFrame) -> None:
    has_income = "annual_income" in df.columns or "income" in df.columns
    assert has_income, "No income column found"
    has_age = "age" in df.columns
    assert has_age, "No age column found"


def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    df = load_data(RAW_PATH)
    validate_data(df)
    clean_df = clean_data(df)
    save_data(clean_df, PROCESSED_PATH)
    print(f"Saved processed data to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
