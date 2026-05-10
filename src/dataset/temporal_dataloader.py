from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import pandas as pd
import torch

from .preprocessing import select_last_event_per_user


@dataclass
class SnapshotBatch:
    sid: int
    events: pd.DataFrame
    prefix_src: torch.Tensor
    prefix_dst: torch.Tensor
    target_src: torch.Tensor
    target_dst: torch.Tensor


def _df_to_edge_tensors(df: Optional[pd.DataFrame], device: torch.device):
    if df is None or len(df) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    src = torch.tensor(df["from"].to_numpy(), dtype=torch.long, device=device)
    dst = torch.tensor(df["to"].to_numpy(), dtype=torch.long, device=device)
    return src, dst


def _shard_events_by_user(events_df: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    if events_df is None or len(events_df) == 0 or world_size == 1:
        return events_df

    per_user = select_last_event_per_user(events_df).sort_values(["from", "timestamp"]).reset_index(drop=True)
    local = per_user.iloc[rank::world_size].reset_index(drop=True)
    return local


class SnapshotDataLoader:
    def __init__(
        self,
        events_by_sid: Dict[int, pd.DataFrame],
        mp_by_sid: Dict[int, pd.DataFrame],
        window_sids: int,
        device: torch.device,
        users_per_batch: int = 0,
        split_by_user_for_ddp: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.events_by_sid = events_by_sid
        self.mp_by_sid = mp_by_sid
        self.window_sids = window_sids
        self.device = device
        self.users_per_batch = users_per_batch
        self.split_by_user_for_ddp = split_by_user_for_ddp
        self.rank = rank
        self.world_size = world_size
        self.sids = sorted(events_by_sid.keys())

    def __len__(self):
        return len(self.sids)

    def __iter__(self) -> Iterator[SnapshotBatch]:
        seen_sids: List[int] = []

        for sid in self.sids:
            full_events_df = self.events_by_sid[sid]

            if self.split_by_user_for_ddp and self.world_size > 1:
                local_events_df = _shard_events_by_user(
                    full_events_df,
                    rank=self.rank,
                    world_size=self.world_size,
                )
            else:
                local_events_df = full_events_df

            if self.window_sids == 0:
                prefix_sids = seen_sids
            else:
                prefix_sids = seen_sids[-self.window_sids:]

            prefix_parts = []
            for prev_sid in prefix_sids:
                mp_df = self.mp_by_sid.get(prev_sid)
                if mp_df is not None and len(mp_df) > 0:
                    prefix_parts.append(mp_df)

            prefix_df = pd.concat(prefix_parts, ignore_index=True) if len(prefix_parts) > 0 else None
            prefix_src, prefix_dst = _df_to_edge_tensors(prefix_df, self.device)

            sid_batches = self._split_events_by_users(local_events_df)

            for batch_events_df in sid_batches:
                target_src, target_dst = _df_to_edge_tensors(batch_events_df, self.device)
                yield SnapshotBatch(
                    sid=int(sid),
                    events=batch_events_df,
                    prefix_src=prefix_src,
                    prefix_dst=prefix_dst,
                    target_src=target_src,
                    target_dst=target_dst,
                )

            seen_sids.append(sid)

    def _split_events_by_users(self, events_df: pd.DataFrame) -> List[pd.DataFrame]:
        if events_df is None or len(events_df) == 0:
            return [events_df]

        # For DDP, keep one step per snapshot on every rank.
        if self.split_by_user_for_ddp:
            return [events_df]

        if self.users_per_batch <= 0:
            return [events_df]

        uniq_users = events_df["from"].drop_duplicates().to_numpy()
        batches = []
        for start in range(0, len(uniq_users), self.users_per_batch):
            batch_users = uniq_users[start : start + self.users_per_batch]
            batches.append(events_df[events_df["from"].isin(batch_users)])

        return batches
