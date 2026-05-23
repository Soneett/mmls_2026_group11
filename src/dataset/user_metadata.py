from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_ml100k_user_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        return {}

    columns = ["user_id", "age", "gender", "occupation", "zip_code"]

    df = pd.read_csv(
        path,
        sep="|",
        names=columns,
        encoding="latin-1",
    )

    metadata: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        user_id = str(row["user_id"])
        metadata[user_id] = {
            "user_id": int(row["user_id"]),
            "age": None if pd.isna(row["age"]) else int(row["age"]),
            "gender": None if pd.isna(row["gender"]) else row["gender"],
            "occupation": None if pd.isna(row["occupation"]) else row["occupation"],
            "zip_code": None if pd.isna(row["zip_code"]) else str(row["zip_code"]),
        }

    return metadata
