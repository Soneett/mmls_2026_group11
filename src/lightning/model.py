import pytorch_lightning as L
import torch

from src.training.runner import init_models, init_optimizer, init_scheduler
from src.training.train_epoch import (
    compute_external_distillation_batch_loss,
    compute_student_ce_batch_loss,
    compute_train_batch_loss,
)
from src.training.evaluation import compute_eval_batch_stats, aggregate_eval_stats
from src.training.state import GraphMeta
from src.config import load_config
from src.training.distillation import load_frozen_teacher


class TemporalLightningModule(L.LightningModule):
    def __init__(self, cfg, num_nodes: int, num_items: int, item_offset: int):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.node_emb, self.encoder, self.compressor = init_models(cfg, num_nodes)
        self.teacher = None

        self.graph_meta = GraphMeta(
            num_nodes=num_nodes,
            num_items=num_items,
            item_offset=item_offset,
        )

        self.val_outputs = []
        self.test_outputs = []

        if self.cfg.training_objective == "external_distillation":
            self._init_frozen_teacher(num_nodes)

    def _init_frozen_teacher(self, num_nodes: int):
        if not self.cfg.teacher_checkpoint_path:
            raise ValueError("teacher_checkpoint_path is required for external_distillation.")

        teacher_cfg = (
            load_config(self.cfg.teacher_config_path)
            if self.cfg.teacher_config_path
            else self.cfg
        )
        teacher = load_frozen_teacher(
            checkpoint_path=self.cfg.teacher_checkpoint_path,
            cfg=teacher_cfg,
            num_nodes=num_nodes,
            map_location="cpu",
        )
        # Keep the frozen teacher out of Lightning checkpoints and optimizers.
        object.__setattr__(self, "teacher", teacher)

    def on_fit_start(self):
        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.freeze()

    def on_save_checkpoint(self, checkpoint):
        # Keep student checkpoints compact; teacher is restored from teacher_checkpoint_path.
        checkpoint["state_dict"] = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if not key.startswith("teacher.")
        }

    def training_step(self, batch, batch_idx):
        if self.cfg.training_objective == "external_distillation":
            out = compute_external_distillation_batch_loss(
                batch=batch,
                encoder=self.encoder,
                compressor=self.compressor,
                node_emb=self.node_emb,
                teacher=self.teacher,
                graph_meta=self.graph_meta,
                kd_weight=self.cfg.kd_weight,
                kd_temperature=self.cfg.kd_temperature,
                teacher_branch=self.cfg.kd_teacher_branch,
                student_branch=self.cfg.kd_student_branch,
                device=self.device,
            )
        elif self.cfg.training_objective == "student_ce":
            out = compute_student_ce_batch_loss(
                batch=batch,
                encoder=self.encoder,
                compressor=self.compressor,
                node_emb=self.node_emb,
                graph_meta=self.graph_meta,
                student_branch=self.cfg.kd_student_branch,
                device=self.device,
            )
        else:
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
            self.log("train_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        self.log("train_loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=out["n_users"])

        if self.cfg.training_objective == "external_distillation":
            self.log("train_loss_ce", out["loss_ce"], on_step=False, on_epoch=True, batch_size=out["n_users"])
            self.log("train_loss_kd", out["loss_kd"], on_step=False, on_epoch=True, batch_size=out["n_users"])
        elif self.cfg.training_objective == "student_ce":
            self.log("train_loss_ce", out["loss_ce"], on_step=False, on_epoch=True, batch_size=out["n_users"])
        else:
            self.log("train_loss_big", out["loss_big"], on_step=False, on_epoch=True, batch_size=out["n_users"])
            self.log("train_loss_small", out["loss_small"], on_step=False, on_epoch=True, batch_size=out["n_users"])
            self.log("train_distill_loss", out["distill_loss"], on_step=False, on_epoch=True, batch_size=out["n_users"])

        return out["loss"]

    def _eval_branches(self):
        if self.cfg.training_objective in {"student_ce", "external_distillation"}:
            return (self.cfg.kd_student_branch,)
        return ("big", "small")

    def _log_eval_metrics(self, metrics, prefix: str):
        if self.cfg.training_objective in {"student_ce", "external_distillation"}:
            student_branch = self.cfg.kd_student_branch
            ndcg = metrics[f"ndcg_{student_branch}"]
            coverage = metrics[f"coverage_{student_branch}"]

            self.log(f"{prefix}_ndcg_student", ndcg, prog_bar=True)
            self.log(f"{prefix}_coverage_student", coverage, prog_bar=True)

            # Backward-compatible aliases used by existing checkpoints/configs.
            self.log(f"{prefix}_ndcg_small", ndcg, prog_bar=False)
            self.log(f"{prefix}_coverage_small", coverage, prog_bar=False)
            return

        self.log(f"{prefix}_ndcg_big", metrics["ndcg_big"], prog_bar=False)
        self.log(f"{prefix}_ndcg_small", metrics["ndcg_small"], prog_bar=True)
        self.log(f"{prefix}_coverage_big", metrics["coverage_big"], prog_bar=False)
        self.log(f"{prefix}_coverage_small", metrics["coverage_small"], prog_bar=True)

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
            eval_branches=self._eval_branches(),
        )
        if stats is not None:
            self.val_outputs.append(stats)

    def on_validation_epoch_end(self):
        metrics = aggregate_eval_stats(
            self.val_outputs,
            self.graph_meta.num_items,
            eval_branches=self._eval_branches(),
        )
        self._log_eval_metrics(metrics, prefix="val")

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
            eval_branches=self._eval_branches(),
        )
        if stats is not None:
            self.test_outputs.append(stats)

    def on_test_epoch_end(self):
        metrics = aggregate_eval_stats(
            self.test_outputs,
            self.graph_meta.num_items,
            eval_branches=self._eval_branches(),
        )
        self._log_eval_metrics(metrics, prefix="test")

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