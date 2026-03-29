import argparse
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.config import load_config
from src.utils.seed import seed_everything
from src.lightning.data import TemporalGraphDataModule
from src.lightning.model import TemporalLightningModule


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

    wandb_logger = WandbLogger(
        project=cfg.project,
        name=cfg.run_name,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{cfg.checkpoint_dir}/{cfg.run_name}",
        filename="best-{epoch}-{val_ndcg_small:.4f}",
        monitor="val_ndcg_small",
        mode="max",
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if "cuda" in cfg.device else ("mps" if cfg.device == "mps" else "cpu"),
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        deterministic=True,
        gradient_clip_val=cfg.grad_clip,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()