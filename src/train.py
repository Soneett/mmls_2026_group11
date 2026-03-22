import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import load_config
from src.utils.seed import seed_everything
from src.lightning.data import TemporalGraphDataModule
from src.lightning.model import TemporalStreamingModule


def main():
    cfg = load_config("configs/base.yaml")
    seed_everything(cfg.seed)

    dm = TemporalGraphDataModule(cfg)
    dm.setup()

    model = TemporalStreamingModule(
        cfg=cfg,
        num_nodes=dm.dataset.num_nodes,
        num_items=dm.dataset.num_items,
        item_offset=dm.dataset.item_offset,
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb["project"],
        name=f"gcn_L{cfg.n_layers}_emb{cfg.embed_dim}_comp{cfg.compressed_dim}_lr{cfg.lr}",
        config=cfg.__dict__,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch}-{val_ndcg:.4f}",
        monitor="val_ndcg",
        mode="max",
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        accelerator="gpu" if "cuda" in cfg.device else "cpu",
        devices=1,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()