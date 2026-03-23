import math
import torch

from ..graph.graph_compose import compute_z_from_edges
from ..dataset.preprocessing import select_last_event_per_user


@torch.no_grad()
def compute_eval_batch_stats(
    batch,
    encoder,
    compressor,
    node_emb,
    graph_meta,
    k,
    device,
):
    if len(batch.events) == 0:
        return None

    z_big, z_small = compute_z_from_edges(
        edge_src=batch.prefix_src,
        edge_dst=batch.prefix_dst,
        num_nodes=graph_meta.num_nodes,
        encoder=encoder,
        compressor=compressor,
        node_emb=node_emb,
        device=device,
    )

    batch_targets = select_last_event_per_user(batch.events)
    if len(batch_targets) == 0:
        return None

    item_ids_global = torch.arange(
        graph_meta.item_offset,
        graph_meta.item_offset + graph_meta.num_items,
        device=device,
        dtype=torch.long,
    )

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

    ndcg_big_sum = 0.0
    ndcg_small_sum = 0.0

    topk_union_big = set()
    topk_union_small = set()

    for i in range(len(users)):
        pos = pos_items[i]

        hits_big = (topk_big[i] == pos).nonzero(as_tuple=False)
        if hits_big.numel() > 0:
            rank_big = hits_big[0].item()
            ndcg_big_sum += 1.0 / math.log2(rank_big + 2)

        hits_small = (topk_small[i] == pos).nonzero(as_tuple=False)
        if hits_small.numel() > 0:
            rank_small = hits_small[0].item()
            ndcg_small_sum += 1.0 / math.log2(rank_small + 2)

        topk_union_big.update(topk_big[i].detach().cpu().tolist())
        topk_union_small.update(topk_small[i].detach().cpu().tolist())

    return {
        "ndcg_big_sum": ndcg_big_sum,
        "ndcg_small_sum": ndcg_small_sum,
        "n_users": int(users.numel()),
        "topk_union_big": topk_union_big,
        "topk_union_small": topk_union_small,
    }


def aggregate_eval_stats(outputs, num_items: int):
    if len(outputs) == 0:
        return {
            "ndcg_big": 0.0,
            "ndcg_small": 0.0,
            "coverage_big": 0.0,
            "coverage_small": 0.0,
        }

    total_ndcg_big = 0.0
    total_ndcg_small = 0.0
    total_users = 0

    topk_union_big = set()
    topk_union_small = set()

    for out in outputs:
        total_ndcg_big += out["ndcg_big_sum"]
        total_ndcg_small += out["ndcg_small_sum"]
        total_users += out["n_users"]

        topk_union_big.update(out["topk_union_big"])
        topk_union_small.update(out["topk_union_small"])

    denom = max(total_users, 1)

    return {
        "ndcg_big": total_ndcg_big / denom,
        "ndcg_small": total_ndcg_small / denom,
        "coverage_big": len(topk_union_big) / num_items,
        "coverage_small": len(topk_union_small) / num_items,
    }