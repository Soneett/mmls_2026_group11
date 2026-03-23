from dataclasses import dataclass
from typing import Dict, Iterator, Optional, List

import pandas as pd
import torch


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
    ):
        self.events_by_sid = events_by_sid
        self.mp_by_sid = mp_by_sid
        self.window_sids = window_sids
        self.device = device
        self.sids = sorted(events_by_sid.keys())

    def __len__(self):
        return len(self.sids)

    def __iter__(self) -> Iterator[SnapshotBatch]:
        seen_sids: List[int] = []

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
            target_src, target_dst = _df_to_edge_tensors(events_df, self.device)

            yield SnapshotBatch(
                sid=int(sid),
                events=events_df,
                prefix_src=prefix_src,
                prefix_dst=prefix_dst,
                target_src=target_src,
                target_dst=target_dst,
            )

            seen_sids.append(sid)