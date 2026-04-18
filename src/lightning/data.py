import pytorch_lightning as L
import torch

from src.dataset.temporal_dataloader import SnapshotDataLoader
from src.dataset.temporal_dataset import build_temporal_graph_dataset


class TemporalGraphDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = None

    def setup(self, stage=None):
        self.dataset = build_temporal_graph_dataset(self.cfg)

    def _dist_info(self):
        if self.trainer is None:
            return 0, 1
        rank = int(getattr(self.trainer, "global_rank", 0))
        world_size = int(getattr(self.trainer, "world_size", 1))
        return rank, world_size

    def _loader(self, events_by_sid, split_by_user_for_ddp: bool, users_per_batch: int):
        rank, world_size = self._dist_info()
        return SnapshotDataLoader(
            events_by_sid=events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=torch.device("cpu"),
            users_per_batch=users_per_batch,
            split_by_user_for_ddp=split_by_user_for_ddp,
            rank=rank,
            world_size=world_size,
        )

    def train_dataloader(self):
        return self._loader(
            events_by_sid=self.dataset.train_events_by_sid,
            split_by_user_for_ddp=self.cfg.parallel_mode == "ddp",
            users_per_batch=self.cfg.users_per_batch,
        )

    def val_dataloader(self):
        return self._loader(
            events_by_sid=self.dataset.val_events_by_sid,
            split_by_user_for_ddp=False,
            users_per_batch=0,
        )

    def test_dataloader(self):
        return self._loader(
            events_by_sid=self.dataset.test_events_by_sid,
            split_by_user_for_ddp=False,
            users_per_batch=0,
        )
