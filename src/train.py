import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

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


def _stage_name(run_name: str) -> str:
    rn = run_name.lower()
    if "teacher" in rn:
        return "teacher"
    if "student" in rn:
        return "student"
    return "student"


def _copy_if_exists(src: str | Path, dst: Path):
    src = Path(src)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _run_benchmark_if_possible(cfg, best_ckpt: str, output_json: Path) -> dict | None:
    if not best_ckpt:
        return None
    cmd = [
        "python",
        "-m",
        "src.benchmark_compression",
        "--config",
        str(getattr(cfg, "_config_path")),
        "--checkpoint",
        best_ckpt,
        "--block-size",
        str(int(getattr(cfg, "benchmark_block_size", 1024))),
        "--repeats",
        str(int(getattr(cfg, "benchmark_repeats", 20))),
        "--warmup",
        str(int(getattr(cfg, "benchmark_warmup", 5))),
        "--output",
        str(output_json),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        return {"error": str(exc), "command": " ".join(cmd)}

    if output_json.exists():
        return json.loads(output_json.read_text(encoding="utf-8"))
    return {"error": "benchmark output json was not created"}


def _log_benchmark_to_wandb(logger: WandbLogger | None, bench_metrics: dict | None):

    if logger is None or bench_metrics is None:
        return

    exp = logger.experiment

    history_payload = {}

    # log all top-level numeric metrics
    for k, v in bench_metrics.items():

        if isinstance(v, (int, float)):
            history_payload[f"benchmark/{k}"] = v

    # log nested per-variant results
    results = bench_metrics.get("results", [])

    if isinstance(results, list):

        for row in results:

            variant = row.get("variant", "unknown")

            for k, v in row.items():

                if k == "variant":
                    continue

                if isinstance(v, (int, float)):
                    history_payload[f"benchmark/{variant}/{k}"] = v

    print("=" * 80)
    print("WANDB BENCH PAYLOAD")
    print("=" * 80)
    print(history_payload)

    if history_payload:

        exp.log(history_payload)

        exp.summary.update(history_payload)

        exp.finish()

def _export_artifacts(cfg, cfg_path: str, best_ckpt: str | None, bench_json: Path | None, logger: WandbLogger | None):
    root = Path("export_artifacts")
    root.mkdir(parents=True, exist_ok=True)
    (root / "teacher").mkdir(exist_ok=True)
    (root / "student").mkdir(exist_ok=True)
    (root / "benchmarks").mkdir(exist_ok=True)
    (root / "configs").mkdir(exist_ok=True)

    stage = _stage_name(cfg.run_name)
    if best_ckpt:
        _copy_if_exists(best_ckpt, root / stage / Path(best_ckpt).name)

    _copy_if_exists(cfg_path, root / "configs" / f"{cfg.run_name}.yaml")

    if bench_json is not None and bench_json.exists():
        _copy_if_exists(bench_json, root / "benchmarks" / f"{cfg.run_name}_benchmark.json")

    if logger is not None:
        summary_path = root / stage / f"{cfg.run_name}_wandb_summary.json"
        summary_path.write_text(json.dumps(dict(logger.experiment.summary), indent=2, default=str), encoding="utf-8")

    zip_path = Path("export_artifacts.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                zf.write(p, p.as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg._config_path = args.config
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
    logger = WandbLogger(project=cfg.project, name=cfg.run_name) if global_rank == 0 else None

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{cfg.checkpoint_dir}/{cfg.run_name}",
        filename="best-{epoch}-{val_ndcg_small:.4f}",
        monitor="val_ndcg_small",
        mode="max",
        save_top_k=1,
        save_last=True,
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
        log_every_n_steps=int(
            getattr(cfg, "log_every_n_steps", 50)
        ),
        precision=getattr(cfg, "precision", "32-true"),
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    best_ckpt = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    bench_json = Path("export_artifacts/benchmarks") / f"{cfg.run_name}_benchmark.json"
    bench_result = _run_benchmark_if_possible(cfg, best_ckpt, bench_json)
    _log_benchmark_to_wandb(logger, bench_result)

    _export_artifacts(cfg, args.config, best_ckpt, bench_json, logger)


if __name__ == "__main__":
    main()
