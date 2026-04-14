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

    def train_dataloader(self):
        return SnapshotDataLoader(
            events_by_sid=self.dataset.train_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=torch.device(self.cfg.device),
            users_per_batch=self.cfg.users_per_batch,
            split_by_user_for_ddp=self.cfg.parallel_mode == "ddp",
        )

    def val_dataloader(self):
        return SnapshotDataLoader(
            events_by_sid=self.dataset.val_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=torch.device(self.cfg.device),
            users_per_batch=0,
            split_by_user_for_ddp=False,
        )

    def test_dataloader(self):
        return SnapshotDataLoader(
            events_by_sid=self.dataset.test_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=torch.device(self.cfg.device),
            users_per_batch=0,
            split_by_user_for_ddp=False,
        )
