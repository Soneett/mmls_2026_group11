import argparse
import os

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

    # Kaggle + DDP: инициализация wandb в каждом воркере часто приводит к блокировкам.
    # Логгер создаём только на global rank 0, остальные процессы логируют через sync_dist.
    # В текущей версии PL у WandbLogger нет флага `rank_zero_only=True`,
    # поэтому делаем эквивалент вручную через переменные ранга процесса.
    global_rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    node_rank = os.environ.get("NODE_RANK", "0")
    if global_rank is not None:
        is_global_zero = int(global_rank) == 0
    else:
        # fallback для окружений, где RANK появляется позже (например, часть notebook-runner'ов)
        is_global_zero = int(local_rank) == 0 and int(node_rank) == 0

    logger = False
    if is_global_zero:
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        wandb_logger_kwargs = {
            "project": cfg.project,
            "name": cfg.run_name,
        }
        try:
            import wandb

            wandb_logger_kwargs["settings"] = wandb.Settings(start_method="thread")
        except Exception:
            pass

        logger = WandbLogger(**wandb_logger_kwargs)

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{cfg.checkpoint_dir}/{cfg.run_name}",
        filename="best-{epoch}-{val_ndcg_small:.4f}",
        monitor="val_ndcg_small",
        mode="max",
        save_top_k=1,
    )

    parallel_mode = getattr(cfg, "parallel_mode", "none")
    devices = getattr(cfg, "devices", 1)
    strategy = "auto"
    if parallel_mode == "ddp" and devices and int(devices) > 1:
        strategy = "ddp_find_unused_parameters_false"

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if "cuda" in cfg.device else ("mps" if cfg.device == "mps" else "cpu"),
        devices=devices,
        strategy=strategy,
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
