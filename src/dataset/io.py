from pathlib import Path

import pandas as pd


def load_ml100k_as_events(
    path: str,
    sep: str = ";",
) -> pd.DataFrame:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path_obj, sep=sep)

    expected_cols = {"user_id", "item_id", "timestamp", "rating"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in dataset: {sorted(missing_cols)}"
        )

    df = df.rename(
        columns={
            "user_id": "from",
            "item_id": "to",
        }
    )

    df = df[["from", "to", "timestamp", "rating"]].copy()

    df["from"] = df["from"].astype(str)
    df["to"] = df["to"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise").astype("int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="raise")

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df