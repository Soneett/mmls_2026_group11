import argparse
import copy
import io
import json
import time
from pathlib import Path

import psutil
import torch

from src.config import load_config
from src.dataset.preprocessing import load_raw_ratings, normalize_and_index_ids
from src.dataset.temporal_dataset import build_temporal_graph_dataset
from src.inference.blockwise_scoring import blockwise_topk_dot_product
from src.lightning.model import TemporalLightningModule
from src.quantization.dynamic_int8 import apply_dynamic_int8_quantization, prepare_model_for_dynamic_int8
from src.quantization.fake_quant import replace_linear_with_fake_quant


def _serialize_size_mb(obj) -> float:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return len(buffer.getbuffer()) / (1024 ** 2)


def _model_state_size_mb(model: TemporalLightningModule) -> float:
    state = {
        "node_emb": model.node_emb.state_dict(),
        "encoder": model.encoder.state_dict(),
        "compressor": model.compressor.state_dict(),
    }
    return _serialize_size_mb(state)


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _load_model(cfg, checkpoint_path: str | None):
    dm_dataset = build_temporal_graph_dataset(cfg)
    model = TemporalLightningModule(
        cfg=cfg,
        num_nodes=dm_dataset.num_nodes,
        num_items=dm_dataset.num_items,
        item_offset=dm_dataset.item_offset,
    )
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    model.eval().to(device)
    return model, dm_dataset, device


def _prepare_eval_batch(cfg, dataset, users_limit: int, device: torch.device):
    raw = load_raw_ratings(cfg.ml100k_path, sep=cfg.sep)
    df, _, _, _ = normalize_and_index_ids(raw)
    users = torch.tensor(df["from"].drop_duplicates().head(users_limit).to_numpy(), dtype=torch.long, device=device)
    item_ids_global = torch.arange(dataset.item_offset, dataset.item_offset + dataset.num_items, dtype=torch.long, device=device)
    return users, item_ids_global


def _score_topk(users_z, items_z, k: int, use_blockwise: bool, block_size: int):
    if use_blockwise:
        return blockwise_topk_dot_product(users_z, items_z, k=k, item_block_size=block_size)
    scores = users_z @ items_z.t()
    topk = torch.topk(scores, k=k, dim=1)
    return topk.values, topk.indices


def _benchmark_variant(name, model, users, item_ids, k, repeats, warmup, block_size, use_blockwise, device):
    node_emb = model.node_emb
    encoder = model.encoder
    compressor = model.compressor

    x = node_emb.weight
    process = psutil.Process()
    rss_baseline = process.memory_info().rss
    max_rss = rss_baseline

    with torch.inference_mode():
        for _ in range(warmup):
            z_big = encoder(x)
            z_small = compressor(z_big)
            users_z = z_small[users]
            items_z = z_small[item_ids]
            _ = _score_topk(users_z, items_z, k=k, use_blockwise=use_blockwise, block_size=block_size)

        latencies = []
        for _ in range(repeats):
            _sync_if_cuda(device)
            start = time.perf_counter()
            z_big = encoder(x)
            z_small = compressor(z_big)
            users_z = z_small[users]
            items_z = z_small[item_ids]
            _ = _score_topk(users_z, items_z, k=k, use_blockwise=use_blockwise, block_size=block_size)
            _sync_if_cuda(device)
            end = time.perf_counter()
            latencies.append(end - start)
            max_rss = max(max_rss, process.memory_info().rss)

    mean_latency_s = float(sum(latencies) / len(latencies))
    return {
        "variant": name,
        "latency_ms": mean_latency_s * 1000.0,
        "throughput_users_per_sec": float(users.shape[0] / mean_latency_s),
        "peak_memory_mb": (max_rss - rss_baseline) / (1024 ** 2),
        "model_size_mb": _model_state_size_mb(model),
        "use_blockwise_scoring": use_blockwise,
    }


def _maybe_teacher_quality_drop(cfg, dataset, student_model, users, item_ids, k, block_size, device):
    if not getattr(cfg, "teacher_checkpoint", ""):
        return None
    teacher_cfg_path = getattr(cfg, "teacher_config", "")
    if not teacher_cfg_path:
        return None

    teacher_cfg = load_config(teacher_cfg_path)
    teacher_model = TemporalLightningModule(
        cfg=teacher_cfg,
        num_nodes=dataset.num_nodes,
        num_items=dataset.num_items,
        item_offset=dataset.item_offset,
    )
    ckpt = torch.load(cfg.teacher_checkpoint, map_location="cpu")
    teacher_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    teacher_model.eval().to(device)

    with torch.inference_mode():
        ts = teacher_model.compressor(teacher_model.encoder(teacher_model.node_emb.weight))
        ss = student_model.compressor(student_model.encoder(student_model.node_emb.weight))
        t_scores, _ = _score_topk(ts[users], ts[item_ids], k=k, use_blockwise=True, block_size=block_size)
        s_scores, _ = _score_topk(ss[users], ss[item_ids], k=k, use_blockwise=True, block_size=block_size)
        t_mean = t_scores.mean().item()
        s_mean = s_scores.mean().item()

    if abs(t_mean) < 1e-12:
        return 0.0
    return max(0.0, (t_mean - s_mean) / abs(t_mean) * 100.0)


def _log_to_wandb(cfg, metrics: dict):
    try:
        import wandb
    except ImportError:
        return

    run = wandb.init(project=cfg.project, name=f"{cfg.run_name}_benchmark", config=metrics.get("config_flags", {}), reinit=True)
    flat = {}
    for row in metrics["results"]:
        prefix = row["variant"]
        flat[f"{prefix}/latency_ms"] = row["latency_ms"]
        flat[f"{prefix}/throughput_users_per_sec"] = row["throughput_users_per_sec"]
        flat[f"{prefix}/peak_memory_mb"] = row["peak_memory_mb"]
        flat[f"{prefix}/model_size_mb"] = row["model_size_mb"]
    if metrics.get("quality_drop_pct_vs_teacher") is not None:
        flat["quality_drop_pct_vs_teacher"] = metrics["quality_drop_pct_vs_teacher"]
    wandb.log(flat)
    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--users-limit", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--output", type=str, default="artifacts/compression_metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, dataset, device = _load_model(cfg, args.checkpoint or None)
    users, item_ids = _prepare_eval_batch(cfg, dataset, users_limit=args.users_limit, device=device)

    rows = []
    fp32_base = copy.deepcopy(model)
    rows.append(_benchmark_variant("fp32", fp32_base, users, item_ids, args.k, args.repeats, args.warmup, args.block_size, False, device))

    if getattr(cfg, "use_blockwise_scoring", True):
        rows.append(_benchmark_variant("blockwise", copy.deepcopy(model), users, item_ids, args.k, args.repeats, args.warmup, args.block_size, True, device))

    if getattr(cfg, "use_fake_quant", False):
        fq_model = copy.deepcopy(model)
        fq_model.encoder = replace_linear_with_fake_quant(fq_model.encoder, inplace=False)
        fq_model.compressor = replace_linear_with_fake_quant(fq_model.compressor, inplace=False)
        rows.append(_benchmark_variant("fake_quant", fq_model, users, item_ids, args.k, args.repeats, args.warmup, args.block_size, True, device))

    if getattr(cfg, "use_dynamic_int8", True):
        int8_model = copy.deepcopy(model)
        if getattr(cfg, "use_fake_quant", False):
            int8_model.encoder = replace_linear_with_fake_quant(int8_model.encoder, inplace=False)
            int8_model.compressor = replace_linear_with_fake_quant(int8_model.compressor, inplace=False)
            int8_model.encoder = prepare_model_for_dynamic_int8(int8_model.encoder, inplace=False)
            int8_model.compressor = prepare_model_for_dynamic_int8(int8_model.compressor, inplace=False)
        else:
            int8_model.encoder = apply_dynamic_int8_quantization(int8_model.encoder, inplace=False)
            int8_model.compressor = apply_dynamic_int8_quantization(int8_model.compressor, inplace=False)
        int8_model.to("cpu")
        rows.append(_benchmark_variant("dynamic_int8", int8_model, users.cpu(), item_ids.cpu(), args.k, args.repeats, args.warmup, args.block_size, True, torch.device("cpu")))

    fp32_latency = next((r["latency_ms"] for r in rows if r["variant"] == "fp32"), None)
    int8_latency = next((r["latency_ms"] for r in rows if r["variant"] == "dynamic_int8"), None)
    blockwise_latency = next((r["latency_ms"] for r in rows if r["variant"] == "blockwise"), None)

    metrics = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "results": rows,
        "fp32_latency_ms": fp32_latency,
        "int8_latency_ms": int8_latency,
        "blockwise_latency_ms": blockwise_latency,
        "speedup_int8": (fp32_latency / int8_latency) if fp32_latency and int8_latency else None,
        "speedup_blockwise": (fp32_latency / blockwise_latency) if fp32_latency and blockwise_latency else None,
        "quality_drop_pct_vs_teacher": _maybe_teacher_quality_drop(cfg, dataset, model, users, item_ids, args.k, args.block_size, device),
        "config_flags": {
            "use_blockwise_scoring": bool(getattr(cfg, "use_blockwise_scoring", True)),
            "use_dynamic_int8": bool(getattr(cfg, "use_dynamic_int8", True)),
            "use_fake_quant": bool(getattr(cfg, "use_fake_quant", False)),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _log_to_wandb(cfg, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
