from dataclasses import dataclass
import math
from typing import Dict, Iterator, Optional, List

import pandas as pd
import torch
import torch.distributed as dist


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


class SnapshotDataLoader:
    def __init__(
        self,
        events_by_sid: Dict[int, pd.DataFrame],
        mp_by_sid: Dict[int, pd.DataFrame],
        window_sids: int,
        device: torch.device,
        users_per_batch: int = 0,
        split_by_user_for_ddp: bool = False,
    ):
        self.events_by_sid = events_by_sid
        self.mp_by_sid = mp_by_sid
        self.window_sids = window_sids
        self.device = device
        self.users_per_batch = users_per_batch
        self.split_by_user_for_ddp = split_by_user_for_ddp
        self.sids = sorted(events_by_sid.keys())

    def __len__(self):
        return len(self.sids)

    def __iter__(self) -> Iterator[SnapshotBatch]:
        seen_sids: List[int] = []
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        for sid in self.sids:
            events_df = self.events_by_sid[sid]

            if self.window_sids == 0:
                prefix_sids = seen_sids
            else:
                prefix_sids = seen_sids[-self.window_sids:]

            prefix_parts = []
            for prev_sid in prefix_sids:
                mp_df = self.mp_by_sid.get(prev_sid)
                if mp_df is not None and len(mp_df) > 0:
                    prefix_parts.append(mp_df)

            prefix_df = (
                pd.concat(prefix_parts, ignore_index=True)
                if len(prefix_parts) > 0
                else None
            )

            prefix_src, prefix_dst = _df_to_edge_tensors(prefix_df, self.device)
            sid_batches = self._split_events_by_users(events_df)
            if self.split_by_user_for_ddp and world_size > 1:
                sid_batches = self._split_batches_evenly_across_ranks(
                    sid_batches=sid_batches,
                    events_df=events_df,
                    world_size=world_size,
                    rank=rank,
                )
                print(f"RANK {rank}: num_batches = {len(sid_batches)}", flush=True)

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
        if self.users_per_batch <= 0:
            return [events_df]

        uniq_users = events_df["from"].drop_duplicates().to_numpy()
        batches = []
        for start in range(0, len(uniq_users), self.users_per_batch):
            batch_users = uniq_users[start:start + self.users_per_batch]
            batches.append(events_df[events_df["from"].isin(batch_users)])

        return batches

    def _split_batches_evenly_across_ranks(
        self,
        sid_batches: List[pd.DataFrame],
        events_df: Optional[pd.DataFrame],
        world_size: int,
        rank: int,
    ) -> List[pd.DataFrame]:
        # Safety: if for some reason no batches exist, create one empty batch.
        if len(sid_batches) == 0:
            sid_batches = [self._empty_events_like(events_df)]

        local_batches = sid_batches[rank::world_size]

        # Every rank must iterate the same number of times to avoid DDP hangs.
        target_num_batches = max(1, math.ceil(len(sid_batches) / world_size))
        if len(local_batches) < target_num_batches:
            dummy = self._empty_events_like(events_df)
            local_batches = local_batches + [dummy] * (target_num_batches - len(local_batches))

        return local_batches

    @staticmethod
    def _empty_events_like(events_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if events_df is None:
            return pd.DataFrame(columns=["from", "to"])
        return events_df.iloc[0:0].copy()
