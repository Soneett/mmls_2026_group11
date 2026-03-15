import pandas as pd
from typing import Dict, Tuple
import numpy as np
import torch

def build_bipartite_id_maps(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:

    user_map = {}
    item_map = {}

    def map_user(x):

        if x not in user_map:
            user_map[x] = len(user_map)

        return user_map[x]

    def map_item(x):

        if x not in item_map:
            item_map[x] = len(item_map)

        return item_map[x]

    df = df.copy()

    df["from"] = df["from"].astype(str).map(map_user)
    df["to"] = df["to"].astype(str).map(map_item)

    item_offset = len(user_map)

    df["to"] = df["to"] + item_offset

    return df, user_map, item_map

def select_last_event_per_user(batch_df: pd.DataFrame, item_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Внутри одного sid берём ровно 1 позитив на пользователя: последний по timestamp.
    Возвращаем:
      users_global: [U]
      pos_items_local: [U]  (0..num_items-1)
    """
    if len(batch_df) == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
    
    g = batch_df.sort_values(["from", "timestamp"], ascending=[True, True])

    last_idx = g.groupby("from", sort=False).tail(1).index
    picked = g.loc[last_idx]

    users_global = torch.tensor(picked["from"].to_numpy(dtype=np.int64), dtype=torch.long)
    pos_items_local = torch.tensor((picked["to"].to_numpy(dtype=np.int64) - item_offset), dtype=torch.long)
    return users_global, pos_items_local
