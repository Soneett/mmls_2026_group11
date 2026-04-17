import pytorch_lightning as L
import torch

from src.training.runner import init_models, init_optimizer, init_scheduler
from src.training.train_epoch import compute_train_batch_loss
from src.training.evaluation import compute_eval_batch_stats, aggregate_eval_stats
from src.training.state import GraphMeta


class TemporalLightningModule(L.LightningModule):
    def __init__(self, cfg, num_nodes: int, num_items: int, item_offset: int):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.node_emb, self.encoder, self.compressor = init_models(cfg, num_nodes)

        self.graph_meta = GraphMeta(
            num_nodes=num_nodes,
            num_items=num_items,
            item_offset=item_offset,
        )

        self.val_outputs = []
        self.test_outputs = []

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def training_step(self, batch, batch_idx):
        out = compute_train_batch_loss(
            batch=batch,
            encoder=self.encoder,
            compressor=self.compressor,
            node_emb=self.node_emb,
            graph_meta=self.graph_meta,
            distillation_weight=self.cfg.distillation_weight,
            device=self.device,
        )

        if out is None:
            loss = self.node_emb.weight.sum() * 0.0
            self.log("train_loss", 0.0, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss

        self.log(
            "train_loss",
            out["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=out["n_users"],
            sync_dist=True,
        )
        self.log("train_loss_big", out["loss_big"], on_step=True, on_epoch=True, batch_size=out["n_users"], sync_dist=True)
        self.log("train_loss_small", out["loss_small"], on_step=True, on_epoch=True, batch_size=out["n_users"], sync_dist=True)
        self.log("train_distill_loss", out["distill_loss"], on_step=True, on_epoch=True, batch_size=out["n_users"], sync_dist=True)

        return out["loss"]

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def validation_step(self, batch, batch_idx):
        stats = compute_eval_batch_stats(
            batch=batch,
            encoder=self.encoder,
            compressor=self.compressor,
            node_emb=self.node_emb,
            graph_meta=self.graph_meta,
            k=self.cfg.k,
            device=self.device,
        )
        if stats is not None:
            self.val_outputs.append(stats)

    def on_validation_epoch_end(self):
        metrics = aggregate_eval_stats(
            self.val_outputs,
            self.graph_meta.num_items,
            device=self.device,
        )

        self.log("val_ndcg_big", metrics["ndcg_big"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_ndcg_small", metrics["ndcg_small"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_coverage_big", metrics["coverage_big"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_coverage_small", metrics["coverage_small"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        stats = compute_eval_batch_stats(
            batch=batch,
            encoder=self.encoder,
            compressor=self.compressor,
            node_emb=self.node_emb,
            graph_meta=self.graph_meta,
            k=self.cfg.k,
            device=self.device,
        )
        if stats is not None:
            self.test_outputs.append(stats)

    def on_test_epoch_end(self):
        metrics = aggregate_eval_stats(
            self.test_outputs,
            self.graph_meta.num_items,
            device=self.device,
        )

        self.log("test_ndcg_big", metrics["ndcg_big"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_ndcg_small", metrics["ndcg_small"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_coverage_big", metrics["coverage_big"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_coverage_small", metrics["coverage_small"], on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = init_optimizer(
            cfg=self.cfg,
            node_emb=self.node_emb,
            encoder=self.encoder,
            compressor=self.compressor,
        )

        if getattr(self.cfg, "use_scheduler", False):
            scheduler = init_scheduler(self.cfg, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
