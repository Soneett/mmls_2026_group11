import torch
import wandb

from .config import CFG
from .utils.seed import seed_everything

from .dataset.temporal_dataset import build_temporal_graph_dataset
from .dataset.temporal_dataloader import SnapshotDataLoader

from .training.runner import init_models_and_opt
from .training.train_epoch import train_epoch_streaming
from .training.evaluation import eval_streaming
from .training.state import GraphMeta


def main():
    cfg = CFG()
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    wandb.init(
        project="dynamic_gnn_recommender_movielens",
        name=f"gcn_L{cfg.n_layers}_emb{cfg.embed_dim}_comp{cfg.compressed_dim}_lr{cfg.lr}",
        config=cfg.__dict__,
    )

    dataset = build_temporal_graph_dataset(cfg)

    train_loader = SnapshotDataLoader(
        events_by_sid=dataset.train_events_by_sid,
        mp_by_sid=dataset.mp_by_sid,
        window_sids=cfg.window_sids,
        device=device,
    )

    val_loader = SnapshotDataLoader(
        events_by_sid=dataset.val_events_by_sid,
        mp_by_sid=dataset.mp_by_sid,
        window_sids=cfg.window_sids,
        device=device,
    )

    test_loader = SnapshotDataLoader(
        events_by_sid=dataset.test_events_by_sid,
        mp_by_sid=dataset.mp_by_sid,
        window_sids=cfg.window_sids,
        device=device,
    )

    state = init_models_and_opt(cfg, dataset.num_nodes, device)

    graph_meta = GraphMeta(
        num_nodes=dataset.num_nodes,
        num_items=dataset.num_items,
        item_offset=dataset.item_offset,
    )

    best_val = 0.0

    for epoch in range(cfg.epochs):
        loss = train_epoch_streaming(
            train_loader=train_loader,
            state=state,
            graph_meta=graph_meta,
            cfg=cfg,
            device=device,
        )

        train_metrics = eval_streaming(
            data_loader=train_loader,
            state=state,
            graph_meta=graph_meta,
            k=cfg.k,
            device=device,
        )

        val_metrics = eval_streaming(
            data_loader=val_loader,
            state=state,
            graph_meta=graph_meta,
            k=cfg.k,
            device=device,
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

    test_metrics = eval_streaming(
        data_loader=test_loader,
        state=state,
        graph_meta=graph_meta,
        k=cfg.k,
        device=device,
    )

    wandb.log({
        "test_ndcg": test_metrics["ndcg_small"],
        "test_coverage": test_metrics["coverage_small"],
    })

    wandb.finish()


if __name__ == "__main__":
    main()