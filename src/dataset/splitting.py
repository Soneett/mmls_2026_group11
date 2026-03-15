import numpy as np
import pandas as pd

def bounds_event_ratio_split(df, train_ratio, val_ratio):

    df_sorted = df.sort_values("timestamp")

    times = df_sorted["timestamp"].to_numpy()

    n = len(times)

    idx_val = int(n * train_ratio)
    idx_test = int(n * (train_ratio + val_ratio))

    val_time = times[idx_val]
    test_time = times[idx_test]

    return val_time, test_time


def gran_to_seconds(gran: str) -> int:
    g = gran.lower()

    if g == "h":
        return 3600

    if g == "d":
        return 86400

    raise ValueError("snapshot_gran must be 'h' or 'd'")


def group_by_sid(df, sid_col="sid"):

    out = {}

    for sid, g in df.groupby(sid_col):
        out[int(sid)] = g.reset_index(drop=True)

    return out
