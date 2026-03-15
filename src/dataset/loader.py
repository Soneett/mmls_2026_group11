import pandas as pd

def load_ml100k_as_events(path: str, sep=";"):

    df = pd.read_csv(
        path,
        sep=sep,
        header=0,
        names=["user_id", "item_id", "timestamp", "rating"],
        engine="python",
    )

    df = df.dropna(subset=["user_id", "item_id", "timestamp"])

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["timestamp"] = df["timestamp"].astype("int64")

    df = df.sort_values("timestamp")

    out = df[["user_id", "item_id", "timestamp"]].rename(
        columns={"user_id": "from", "item_id": "to"}
    )

    return out
