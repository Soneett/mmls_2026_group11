import argparse
import os

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from src.config import load_config
from src.lightning.data import TemporalGraphDataModule
from src.lightning.model import TemporalLightningModule
from src.utils.seed import seed_everything


def _build_strategy(cfg, accelerator: str):
    if accelerator != "gpu":
        return "auto"

    parallel_mode = getattr(cfg, "parallel_mode", "none")
    devices = int(getattr(cfg, "devices", 1))

    if parallel_mode == "ddp" and devices > 1:
        return "ddp_find_unused_parameters_false"

    if parallel_mode in {"zero1", "deepspeed_zero1"}:
        return DeepSpeedStrategy(
            stage=int(getattr(cfg, "zero_stage", 1)),
            offload_optimizer=bool(getattr(cfg, "zero_offload_optimizer", False)),
            offload_optimizer_device=getattr(cfg, "zero_offload_optimizer_device", "cpu"),
        )

    return "auto"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    dm = TemporalGraphDataModule(cfg)
    dm.setup()

    model = TemporalLightningModule(
        cfg=cfg,
        num_nodes=dm.dataset.num_nodes,
        num_items=dm.dataset.num_items,
        item_offset=dm.dataset.item_offset,
    )

    global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    logger = None
    if global_rank == 0:
        logger = WandbLogger(project=cfg.project, name=cfg.run_name)

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{cfg.checkpoint_dir}/{cfg.run_name}",
        filename="best-{epoch}-{val_ndcg_small:.4f}",
        monitor="val_ndcg_small",
        mode="max",
        save_top_k=1,
    )

    accelerator = "gpu" if torch.cuda.is_available() and "cuda" in cfg.device else "cpu"
    strategy = _build_strategy(cfg, accelerator)

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=int(getattr(cfg, "devices", 1)),
        num_nodes=int(getattr(cfg, "num_nodes", 1)),
        strategy=strategy,
        use_distributed_sampler=False,
        logger=logger,
        callbacks=[checkpoint_cb],
        deterministic=True,
        gradient_clip_val=cfg.grad_clip,
        accumulate_grad_batches=max(1, int(getattr(cfg, "grad_accum_steps", 1))),
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
