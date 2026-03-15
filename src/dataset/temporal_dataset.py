from dataclasses import dataclass
import pandas as pd

from .io import load_ml100k_as_events
from .preprocessing import (
    build_bipartite_id_maps,
    bounds_event_ratio_split,
    gran_to_seconds,
    group_by_sid,
)


@dataclass
class TemporalGraphDataset:
    df: pd.DataFrame
    train_events_by_sid: dict
    val_events_by_sid: dict
    test_events_by_sid: dict
    mp_by_sid: dict

    num_users: int
    num_items: int
    num_nodes: int
    item_offset: int
    val_time: int
    test_time: int


def build_temporal_graph_dataset(cfg) -> TemporalGraphDataset:
    df = load_ml100k_as_events(cfg.ml100k_path)

    df, user_map, item_map = build_bipartite_id_maps(df)

    num_users = len(user_map)
    num_items = len(item_map)
    item_offset = num_users
    num_nodes = num_users + num_items

    val_time, test_time = bounds_event_ratio_split(
        df,
        cfg.train_ratio,
        cfg.val_ratio,
    )

    df["split"] = "train"
    df.loc[df["timestamp"] >= val_time, "split"] = "val"
    df.loc[df["timestamp"] >= test_time, "split"] = "test"

    bin_sec = gran_to_seconds(cfg.snapshot_gran)
    df["sid"] = (df["timestamp"] // bin_sec).astype("int64")

    df_events = df[["from", "to", "timestamp", "sid", "split"]].copy()

    df_rev = df_events.copy()
    df_rev[["from", "to"]] = df_rev[["to", "from"]]

    df_mp = pd.concat([df_events, df_rev], ignore_index=True)
    df_mp = (
        df_mp
        .drop_duplicates(subset=["from", "to", "timestamp"])
        .sort_values(["sid", "timestamp"])
        .reset_index(drop=True)
    )

    train_events_by_sid = group_by_sid(df_events[df_events["split"] == "train"])
    val_events_by_sid = group_by_sid(df_events[df_events["split"] == "val"])
    test_events_by_sid = group_by_sid(df_events[df_events["split"] == "test"])
    mp_by_sid = group_by_sid(df_mp)

    return TemporalGraphDataset(
        df=df,
        train_events_by_sid=train_events_by_sid,
        val_events_by_sid=val_events_by_sid,
        test_events_by_sid=test_events_by_sid,
        mp_by_sid=mp_by_sid,
        num_users=num_users,
        num_items=num_items,
        num_nodes=num_nodes,
        item_offset=item_offset,
        val_time=val_time,
        test_time=test_time,
    )