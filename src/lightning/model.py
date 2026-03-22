import lightning as L
import torch

from src.training.runner import init_models_and_opt
from src.training.train_epoch import train_epoch_streaming
from src.training.evaluation import eval_streaming
from src.training.state import GraphMeta


class TemporalStreamingModule(L.LightningModule):
    def __init__(self, cfg, num_nodes, num_items, item_offset):
        super().__init__()
        self.cfg = cfg

        # Важно: trainer не будет сам делать optimizer.step()
        self.automatic_optimization = False

        self.state = init_models_and_opt(
            cfg=cfg,
            num_nodes=num_nodes,
            device=torch.device(cfg.device),
        )

        # чтобы Lightning видел модули как submodules
        self.encoder = self.state.encoder
        self.compressor = self.state.compressor

        self.graph_meta = GraphMeta(
            num_nodes=num_nodes,
            num_items=num_items,
            item_offset=item_offset,
        )

        self.best_val = float("-inf")

    def configure_optimizers(self):
        # Можно вернуть optimizer, чтобы он попал в checkpoint Lightning
        return self.state.optimizer

    def training_step(self, batch, batch_idx):
        # Ничего не делаем: реальное обучение идет в hook ниже
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def on_train_epoch_start(self):
        dm = self.trainer.datamodule

        loss = train_epoch_streaming(
            train_loader=dm.stream_train_loader,
            state=self.state,
            graph_meta=self.graph_meta,
            cfg=self.cfg,
            device=torch.device(self.cfg.device),
        )

        train_metrics = eval_streaming(
            data_loader=dm.stream_train_loader,
            state=self.state,
            graph_meta=self.graph_meta,
            k=self.cfg.k,
            device=torch.device(self.cfg.device),
        )

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_ndcg", train_metrics["ndcg_small"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_coverage", train_metrics["coverage_small"], prog_bar=False, on_step=False, on_epoch=True)
        self.log("lr", self.state.optimizer.param_groups[0]["lr"], prog_bar=False, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self):
        # защита от лишней sanity validation
        if self.trainer.sanity_checking:
            return

        dm = self.trainer.datamodule

        val_metrics = eval_streaming(
            data_loader=dm.stream_val_loader,
            state=self.state,
            graph_meta=self.graph_meta,
            k=self.cfg.k,
            device=torch.device(self.cfg.device),
        )

        val_ndcg = val_metrics["ndcg_small"]
        val_cov = val_metrics["coverage_small"]

        self.log("val_ndcg", val_ndcg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_coverage", val_cov, prog_bar=True, on_step=False, on_epoch=True)

        # если хочешь пока оставить свое ручное сохранение
        if val_ndcg > self.best_val:
            self.best_val = val_ndcg

    def on_test_epoch_start(self):
        dm = self.trainer.datamodule

        test_metrics = eval_streaming(
            data_loader=dm.stream_test_loader,
            state=self.state,
            graph_meta=self.graph_meta,
            k=self.cfg.k,
            device=torch.device(self.cfg.device),
        )

        self.log("test_ndcg", test_metrics["ndcg_small"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_coverage", test_metrics["coverage_small"], prog_bar=True, on_step=False, on_epoch=True)