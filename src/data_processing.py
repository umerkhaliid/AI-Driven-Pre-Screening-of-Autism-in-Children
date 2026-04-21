import pandas as pd
import numpy as np
from src.config import RAW_DATASET_PATH, PROCESSED_TRAIN_PATH, PROCESSED_DATA_DIR


Q_COLS = [f"a{i}" for i in range(1, 25)]


def load_raw_dataset() -> pd.DataFrame:
    return pd.read_csv(RAW_DATASET_PATH)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for c in df.columns
    ]
    return df


def preprocess_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Validate target ----
    if "risk_class" not in df.columns:
        raise ValueError("Target column 'risk_class' not found")

    df["risk_class"] = pd.to_numeric(df["risk_class"], errors="coerce")
    df = df.dropna(subset=["risk_class"])
    df["risk_class"] = df["risk_class"].astype(int)

    # ---- Question columns (a1-a24) ----
    for c in Q_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing question column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Age ----
    if "age_mons" not in df.columns:
        raise ValueError("Missing column: age_mons")
    df["age_mons"] = pd.to_numeric(df["age_mons"], errors="coerce")

    # ---- Gender (stored as internal `sex` for model compatibility) ----
    gender_col = "gender" if "gender" in df.columns else "sex"
    if gender_col not in df.columns:
        raise ValueError("Missing column: gender")
    df["sex"] = pd.to_numeric(df[gender_col], errors="coerce")

    # ---- Jaundice ----
    if "jaundice" not in df.columns:
        raise ValueError("Missing column: jaundice")
    df["jaundice"] = pd.to_numeric(df["jaundice"], errors="coerce")

    # ---- Family history ----
    if "family_mem_with_asd" not in df.columns:
        raise ValueError("Missing column: family_mem_with_asd")
    df["family_mem_with_asd"] = pd.to_numeric(df["family_mem_with_asd"], errors="coerce")

    # ---- Drop columns that cause data leakage or are not needed ----
    drop_cols = []
    for col in ["screening_score", "ethnicity"]:
        if col in df.columns:
            drop_cols.append(col)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # ---- Drop rows with missing core fields ----
    core_cols = Q_COLS + ["age_mons", "sex", "jaundice", "family_mem_with_asd", "risk_class"]
    df = df.dropna(subset=core_cols)

    # ---- Ensure correct data types ----
    df[Q_COLS] = df[Q_COLS].astype(int)
    df["age_mons"] = df["age_mons"].astype(int)
    df["sex"] = df["sex"].astype(int)
    df["jaundice"] = df["jaundice"].astype(int)
    df["family_mem_with_asd"] = df["family_mem_with_asd"].astype(int)
    df["risk_class"] = df["risk_class"].astype(int)

    return df


def save_processed_dataset(df: pd.DataFrame):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_TRAIN_PATH, index=False)


def run_pipeline():
    df = load_raw_dataset()
    df = clean_column_names(df)
    df = preprocess_for_training(df)
    save_processed_dataset(df)

    print("Processed dataset saved to:", PROCESSED_TRAIN_PATH)
    print("Shape:", df.shape)
    print("\nTarget distribution:")
    print(df["risk_class"].value_counts().sort_index())


if __name__ == "__main__":
    run_pipeline()
