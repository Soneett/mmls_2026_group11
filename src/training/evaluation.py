import torch
import numpy as np

from ..graph.graph_compose import compute_z_from_edges
from ..dataset.preprocessing import select_last_event_per_user


@torch.no_grad()
def eval_streaming(
    data_loader,
    state,
    graph_meta,
    k,
    device,
):
    state.encoder.eval()
    state.compressor.eval()
    state.node_emb.eval()

    item_ids_global = torch.arange(
        graph_meta.item_offset,
        graph_meta.item_offset + graph_meta.num_items,
        device=device,
        dtype=torch.long,
    )

    ndcg_big = []
    ndcg_small = []

    topk_union_big = set()
    topk_union_small = set()

    for batch in data_loader:
        if len(batch.events) == 0:
            continue

        z_big, z_small = compute_z_from_edges(
            edge_src=batch.prefix_src,
            edge_dst=batch.prefix_dst,
            num_nodes=graph_meta.num_nodes,
            encoder=state.encoder,
            compressor=state.compressor,
            node_emb=state.node_emb,
            device=device,
        )

        batch_targets = select_last_event_per_user(batch.events)

        if len(batch_targets) == 0:
            continue

        users = torch.tensor(
            batch_targets["from"].to_numpy(),
            dtype=torch.long,
            device=device,
        )

        pos_items = torch.tensor(
            batch_targets["to"].to_numpy() - graph_meta.item_offset,
            dtype=torch.long,
            device=device,
        )

        logits_big = z_big[users] @ z_big[item_ids_global].t()
        logits_small = z_small[users] @ z_small[item_ids_global].t()

        topk_big = torch.topk(logits_big, k=k, dim=1).indices
        topk_small = torch.topk(logits_small, k=k, dim=1).indices

        for i in range(len(users)):
            pos = pos_items[i]

            hits_big = (topk_big[i] == pos).nonzero(as_tuple=False)
            if hits_big.numel() == 0:
                ndcg_big.append(0.0)
            else:
                rank_big = hits_big[0].item()
                ndcg_big.append(1.0 / np.log2(rank_big + 2))

            topk_union_big.update(topk_big[i].cpu().tolist())

            hits_small = (topk_small[i] == pos).nonzero(as_tuple=False)
            if hits_small.numel() == 0:
                ndcg_small.append(0.0)
            else:
                rank_small = hits_small[0].item()
                ndcg_small.append(1.0 / np.log2(rank_small + 2))

            topk_union_small.update(topk_small[i].cpu().tolist())

    coverage_big = len(topk_union_big) / graph_meta.num_items
    coverage_small = len(topk_union_small) / graph_meta.num_items

    return {
        "ndcg_big": float(np.mean(ndcg_big)) if len(ndcg_big) > 0 else 0.0,
        "ndcg_small": float(np.mean(ndcg_small)) if len(ndcg_small) > 0 else 0.0,
        "coverage_big": coverage_big,
        "coverage_small": coverage_small,
    }