import pytorch_lightning as L
import torch
from types import SimpleNamespace

from src.training.runner import init_models, init_optimizer, init_scheduler
from src.training.train_epoch import compute_train_batch_loss
from src.training.evaluation import compute_eval_batch_stats, aggregate_eval_stats
from src.training.state import GraphMeta
from src.config import load_config


class TemporalLightningModule(L.LightningModule):
    def __init__(self, cfg, num_nodes: int, num_items: int, item_offset: int):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.node_emb, self.encoder, self.compressor = init_models(cfg, num_nodes)
        self.teacher_bundle = None
        if getattr(cfg, "distillation_mode", "joint") == "offline_kd":
            self.teacher_bundle = self._load_teacher_bundle(cfg, num_nodes)

        self.graph_meta = GraphMeta(
            num_nodes=num_nodes,
            num_items=num_items,
            item_offset=item_offset,
        )

        self.val_outputs = []
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        teacher_outputs = None
        if self.teacher_bundle is not None:
            with torch.no_grad():
                teacher_outputs = self._forward_teacher(batch)

        out = compute_train_batch_loss(
            batch=batch,
            encoder=self.encoder,
            compressor=self.compressor,
            node_emb=self.node_emb,
            graph_meta=self.graph_meta,
            distillation_mode=getattr(self.cfg, "distillation_mode", "joint"),
            distillation_weight=self.cfg.distillation_weight,
            lambda_kd=getattr(self.cfg, "lambda_kd", 0.0),
            kd_temperature=getattr(self.cfg, "kd_temperature", 1.0),
            teacher_outputs=teacher_outputs,
            device=self.device,
        )

        if out is None:
            loss = self.node_emb.weight.sum() * 0.0
            self.log("train_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        self.log("train_loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=out["n_users"])
        self.log("train_loss_big", out["loss_big"], on_step=False, on_epoch=True, batch_size=out["n_users"])
        self.log("train_loss_small", out["loss_small"], on_step=False, on_epoch=True, batch_size=out["n_users"])
        self.log("train_distill_loss", out["distill_loss"], on_step=False, on_epoch=True, batch_size=out["n_users"])

        return out["loss"]

    def _load_teacher_bundle(self, cfg, num_nodes: int):
        if not getattr(cfg, "teacher_checkpoint", ""):
            raise ValueError("teacher_checkpoint must be set for offline_kd mode")
        if not getattr(cfg, "teacher_config", ""):
            raise ValueError("teacher_config must be set for offline_kd mode")

        teacher_cfg = load_config(cfg.teacher_config)
        teacher_node_emb, teacher_encoder, teacher_compressor = init_models(teacher_cfg, num_nodes)
        ckpt = torch.load(cfg.teacher_checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        teacher_node_emb.load_state_dict({
            k.replace("node_emb.", "", 1): v
            for k, v in state.items()
            if k.startswith("node_emb.")
        })
        teacher_encoder.load_state_dict({
            k.replace("encoder.", "", 1): v
            for k, v in state.items()
            if k.startswith("encoder.")
        })
        teacher_compressor.load_state_dict({
            k.replace("compressor.", "", 1): v
            for k, v in state.items()
            if k.startswith("compressor.")
        })

        teacher_node_emb.requires_grad_(False).eval()
        teacher_encoder.requires_grad_(False).eval()
        teacher_compressor.requires_grad_(False).eval()
        return SimpleNamespace(node_emb=teacher_node_emb, encoder=teacher_encoder, compressor=teacher_compressor)

    def _forward_teacher(self, batch):
        from src.graph.graph_compose import compute_z_from_edges
        from src.dataset.preprocessing import select_last_event_per_user

        teacher = self.teacher_bundle

        z_big, _ = compute_z_from_edges(
            edge_src=batch.prefix_src,
            edge_dst=batch.prefix_dst,
            num_nodes=self.graph_meta.num_nodes,
            encoder=teacher.encoder,
            compressor=teacher.compressor,
            node_emb=teacher.node_emb,
            device=self.device,
        )

        batch_targets = select_last_event_per_user(batch.events)
        users = torch.tensor(batch_targets["from"].to_numpy(), dtype=torch.long, device=self.device)
        item_ids_global = torch.arange(
            self.graph_meta.item_offset,
            self.graph_meta.item_offset + self.graph_meta.num_items,
            device=self.device,
            dtype=torch.long,
        )
        users_z = z_big[users]
        items_z = z_big[item_ids_global]
        logits = users_z @ items_z.t()
        return {"logits": logits}

    def on_fit_start(self):
        if self.teacher_bundle is not None:
            self.teacher_bundle.node_emb.to(self.device)
            self.teacher_bundle.encoder.to(self.device)
            self.teacher_bundle.compressor.to(self.device)

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
        metrics = aggregate_eval_stats(self.val_outputs, self.graph_meta.num_items)

        self.log("val_ndcg_big", metrics["ndcg_big"], prog_bar=False)
        self.log("val_ndcg_small", metrics["ndcg_small"], prog_bar=True)
        self.log("val_coverage_big", metrics["coverage_big"], prog_bar=False)
        self.log("val_coverage_small", metrics["coverage_small"], prog_bar=True)

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
        metrics = aggregate_eval_stats(self.test_outputs, self.graph_meta.num_items)

        self.log("test_ndcg_big", metrics["ndcg_big"])
        self.log("test_ndcg_small", metrics["ndcg_small"])
        self.log("test_coverage_big", metrics["coverage_big"])
        self.log("test_coverage_small", metrics["coverage_small"])

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
