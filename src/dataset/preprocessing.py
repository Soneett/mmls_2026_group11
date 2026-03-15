from typing import Dict, Tuple

import pandas as pd


def build_bipartite_id_maps(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Map users and items into a single shared node index space.

    Users:
        0 ... num_users - 1

    Items:
        num_users ... num_users + num_items - 1
    """
    if not {"from", "to"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'from' and 'to' columns.")

    user_map: Dict[str, int] = {}
    item_map: Dict[str, int] = {}

    def map_user(x: str) -> int:
        if x not in user_map:
            user_map[x] = len(user_map)
        return user_map[x]

    def map_item(x: str) -> int:
        if x not in item_map:
            item_map[x] = len(item_map)
        return item_map[x]

    out = df.copy()

    out["from"] = out["from"].map(map_user)
    out["to"] = out["to"].map(map_item)

    item_offset = len(user_map)
    out["to"] = out["to"] + item_offset

    return out, user_map, item_map


def bounds_event_ratio_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[int, int]:
    """
    Compute timestamp cutoffs for temporal train/val/test split.

    Returns:
        val_time, test_time
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column.")

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1).")

    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0, 1).")

    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    ts = df["timestamp"].sort_values().to_numpy()
    n = len(ts)

    if n == 0:
        raise ValueError("Empty dataframe passed to split function.")

    val_idx = int(n * train_ratio)
    test_idx = int(n * (train_ratio + val_ratio))

    val_idx = min(max(val_idx, 0), n - 1)
    test_idx = min(max(test_idx, 0), n - 1)

    val_time = int(ts[val_idx])
    test_time = int(ts[test_idx])

    return val_time, test_time


def assign_split_by_time(
    df: pd.DataFrame,
    val_time: int,
    test_time: int,
) -> pd.DataFrame:
    """
    Add temporal split column: train / val / test.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column.")

    out = df.copy()
    out["split"] = "train"
    out.loc[out["timestamp"] >= val_time, "split"] = "val"
    out.loc[out["timestamp"] >= test_time, "split"] = "test"

    return out


def gran_to_seconds(gran: str) -> int:
    """
    Convert snapshot granularity string to seconds.
    """
    mapping = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 7 * 86400,
    }

    if gran not in mapping:
        raise ValueError(f"Unsupported granularity: {gran}")

    return mapping[gran]


def assign_snapshot_ids(
    df: pd.DataFrame,
    snapshot_gran: str,
) -> pd.DataFrame:
    """
    Add snapshot id column 'sid' using timestamp binning.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column.")

    out = df.copy()
    bin_sec = gran_to_seconds(snapshot_gran)
    out["sid"] = (out["timestamp"] // bin_sec).astype("int64")

    return out


def make_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns needed for event stream.
    """
    required = {"from", "to", "timestamp", "sid", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for events df: {sorted(missing)}")

    return df[["from", "to", "timestamp", "sid", "split"]].copy()


def make_mirrored_events(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Build bidirectional message-passing edges from directed events.
    """
    required = {"from", "to", "timestamp", "sid", "split"}
    missing = required - set(df_events.columns)
    if missing:
        raise ValueError(
            f"Missing columns for mirrored events: {sorted(missing)}"
        )

    df_rev = df_events.copy()
    df_rev[["from", "to"]] = df_rev[["to", "from"]]

    df_mp = pd.concat([df_events, df_rev], ignore_index=True)
    df_mp = (
        df_mp
        .drop_duplicates(subset=["from", "to", "timestamp"])
        .sort_values(["sid", "timestamp"])
        .reset_index(drop=True)
    )

    return df_mp


def group_by_sid(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Group dataframe into dict:
        sid -> dataframe for this snapshot
    """
    if "sid" not in df.columns:
        raise ValueError("DataFrame must contain 'sid' column.")

    groups = {}
    for sid, g in df.groupby("sid", sort=True):
        groups[int(sid)] = g.sort_values("timestamp").reset_index(drop=True)

    return groups


def select_last_event_per_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the last event per user within the dataframe.
    Useful when each user should contribute one target event.
    """
    required = {"from", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for last-event selection: {sorted(missing)}"
        )

    out = (
        df.sort_values(["from", "timestamp"])
        .groupby("from", as_index=False)
        .tail(1)
        .sort_values(["timestamp", "from"])
        .reset_index(drop=True)
    )

    return out