import pytorch_lightning as L
import torch

from src.dataset.temporal_dataset import build_temporal_graph_dataset
from src.dataset.temporal_dataloader import SnapshotDataLoader


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

    def train_dataloader(self):
        rank, world_size = self._dist_info()
        return SnapshotDataLoader(
            events_by_sid=self.dataset.train_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=torch.device(self.cfg.device),
            users_per_batch=self.cfg.users_per_batch,
            split_by_user_for_ddp=self.cfg.parallel_mode == "ddp",
            device=torch.device("cpu"),
            rank=rank,
            world_size=world_size,
        )

    def val_dataloader(self):
        rank, world_size = self._dist_info()
        return SnapshotDataLoader(
            events_by_sid=self.dataset.val_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=torch.device(self.cfg.device),
            users_per_batch=0,
            split_by_user_for_ddp=False,
            device=torch.device("cpu"),
            rank=rank,
            world_size=world_size,
        )

    def test_dataloader(self):
        rank, world_size = self._dist_info()
        return SnapshotDataLoader(
            events_by_sid=self.dataset.test_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=torch.device(self.cfg.device),
            users_per_batch=0,
            split_by_user_for_ddp=False,
            device=torch.device("cpu"),
            rank=rank,
            world_size=world_size,
        )
