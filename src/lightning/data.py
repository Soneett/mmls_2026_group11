import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset.temporal_dataset import build_temporal_graph_dataset
from src.dataset.temporal_dataloader import SnapshotDataLoader


class _OneBatchDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return 0


class TemporalGraphDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = None

        self.stream_train_loader = None
        self.stream_val_loader = None
        self.stream_test_loader = None

        self._dummy_ds = _OneBatchDataset()

    def setup(self, stage=None):
        device = torch.device(self.cfg.device)

        self.dataset = build_temporal_graph_dataset(self.cfg)

        self.stream_train_loader = SnapshotDataLoader(
            events_by_sid=self.dataset.train_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=device,
        )

        self.stream_val_loader = SnapshotDataLoader(
            events_by_sid=self.dataset.val_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=device,
        )

        self.stream_test_loader = SnapshotDataLoader(
            events_by_sid=self.dataset.test_events_by_sid,
            mp_by_sid=self.dataset.mp_by_sid,
            window_sids=self.cfg.window_sids,
            device=device,
        )

    def train_dataloader(self):
        return DataLoader(self._dummy_ds, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self._dummy_ds, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self._dummy_ds, batch_size=1)