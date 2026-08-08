from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLUMNS = {"label", "statement"}


 # Loads dataset for fake news detection

def resolve_dataset_file(dataset_dir: str | Path, preferred_file: Optional[str] = None) -> Path:
    dataset_dir = Path(dataset_dir)
    candidates = []
    if preferred_file:
        candidates.append(dataset_dir / preferred_file)

    candidates.extend(
        [
            dataset_dir / "phase1_balanced_sample.csv",
            dataset_dir / "test_clean.csv",
            dataset_dir / "valid_clean.csv",
            dataset_dir / "train_clean.csv",
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No dataset CSV found in {dataset_dir}. Place one of: "
        "train_clean.csv, valid_clean.csv, test_clean.csv, phase1_balanced_sample.csv"
    )


def load_claim_dataset(dataset_dir: str | Path, preferred_file: Optional[str] = None) -> pd.DataFrame:
    csv_path = resolve_dataset_file(dataset_dir, preferred_file=preferred_file)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset file {csv_path} is missing required columns: {sorted(missing)}")

    df = df[["statement", "label"]].dropna().copy()
    df["statement"] = df["statement"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["statement"] != ""]
    return df.reset_index(drop=True)
