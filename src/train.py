import torch
import wandb
import pandas as pd

from .config import CFG
from .utils.seed import seed_everything

from .dataset.loader import load_ml100k_as_events
from .dataset.preprocessing import build_bipartite_id_maps, bounds_event_ratio_split, group_by_sid, gran_to_seconds

from .training.runner import init_models_and_opt
from .training.train_epoch import train_epoch_streaming
from .training.evaluation import eval_streaming
from .training.state import GraphMeta

def main():

    cfg = CFG()

    seed_everything(cfg.seed)

    device = torch.device(cfg.device)

    project_name = "dynamic_gnn_recommender_movielens"

    run_name = (
        f"gcn_L{cfg.n_layers}"
        f"_emb{cfg.embed_dim}"
        f"_comp{cfg.compressed_dim}"
        f"_lr{cfg.lr}"
    )

    wandb.init(
        project=project_name,
        name=run_name,
        config=cfg.__dict__,
    )

    print("Loading dataset...")

    df = load_ml100k_as_events(cfg.ml100k_path)

    df, user_map, item_map = build_bipartite_id_maps(df)

    num_users = len(user_map)
    num_items = len(item_map)

    item_offset = num_users
    num_nodes = num_users + num_items

    print(f"Users: {num_users}, Items: {num_items}, Nodes: {num_nodes}")

    val_time, test_time = bounds_event_ratio_split(
        df,
        cfg.train_ratio,
        cfg.val_ratio,
    )

    df["split"] = "train"

    df.loc[df.timestamp >= val_time, "split"] = "val"
    df.loc[df.timestamp >= test_time, "split"] = "test"

    print("Building snapshot ids...")

    bin_sec = gran_to_seconds(cfg.snapshot_gran)

    df["sid"] = (df["timestamp"] // bin_sec).astype("int64")

    df_events = df[["from", "to", "timestamp", "sid", "split"]].copy()

    df_rev = df_events.copy()
    df_rev[["from", "to"]] = df_rev[["to", "from"]]

    df_mp = pd.concat([df_events, df_rev], ignore_index=True)

    df_mp = df_mp.drop_duplicates(
        subset=["from", "to", "timestamp"]
    ).sort_values(["sid", "timestamp"])

    train_events_by_sid = group_by_sid(
        df_events[df_events["split"] == "train"]
    )

    val_events_by_sid = group_by_sid(
        df_events[df_events["split"] == "val"]
    )

    test_events_by_sid = group_by_sid(
        df_events[df_events["split"] == "test"]
    )

    mp_by_sid = group_by_sid(df_mp)

    print(
        f"Train sids: {len(train_events_by_sid)}, "
        f"Val sids: {len(val_events_by_sid)}, "
        f"Test sids: {len(test_events_by_sid)}"
    )

    state = init_models_and_opt(
        cfg,
        num_nodes,
        device
    )

    graph_meta = GraphMeta(
        num_nodes=num_nodes,
        num_items=num_items,
        item_offset=item_offset
    )

    best_val = 0

    print("Start training")

    for epoch in range(cfg.epochs):

        loss = train_epoch_streaming(
            train_events_by_sid,
            mp_by_sid,
            cfg.window_sids,
            state,
            graph_meta,
            cfg,
            device,
        )

        train_metrics = eval_streaming(
            train_events_by_sid,
            mp_by_sid,
            0,
            cfg.window_sids,
            state,
            graph_meta,
            cfg.k,
            device,
        )

        start_prefix_sid = max(train_events_by_sid.keys()) + 1

        val_metrics = eval_streaming(
            val_events_by_sid,
            mp_by_sid,
            start_prefix_sid,
            cfg.window_sids,
            state,
            graph_meta,
            cfg.k,
            device,
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": loss,
            "train_ndcg": train_metrics["ndcg_small"],
            "val_ndcg": val_metrics["ndcg_small"],
            "train_coverage": train_metrics["coverage_small"],
            "val_coverage": val_metrics["coverage_small"],
            "lr": state.optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:03d} | "
            f"loss={loss:.4f} | "
            f"train_ndcg={train_metrics['ndcg_small']:.4f} | "
            f"val_ndcg={val_metrics['ndcg_small']:.4f}"
        )

        if val_metrics["ndcg_small"] > best_val:

            best_val = val_metrics["ndcg_small"]

            torch.save(
                {
                    "encoder": state.encoder.state_dict(),
                    "compressor": state.compressor.state_dict(),
                    "optimizer": state.optimizer.state_dict(),
                    "epoch": epoch,
                },
                "checkpoints/best_model.pt",
            )

    print("Training finished")

    print("Running final test evaluation")

    start_prefix_sid = max(train_events_by_sid.keys()) + 1

    test_metrics = eval_streaming(
        test_events_by_sid,
        mp_by_sid,
        start_prefix_sid,
        cfg.window_sids,
        state,
        graph_meta,
        cfg.k,
        device,
    )

    print(
        f"TEST NDCG@{cfg.k}: {test_metrics['ndcg_small']:.4f}"
    )

    wandb.log({
        "test_ndcg": test_metrics["ndcg_small"],
        "test_coverage": test_metrics["coverage_small"],
    })

    wandb.finish()


if __name__ == "__main__":
    main()