import math
import torch

from ..graph.graph_compose import compute_z_from_edges
from ..dataset.preprocessing import select_last_event_per_user
from .distillation import BranchOutputs, logits_from_embeddings


def _empty_metrics(eval_branches):
    metrics = {}
    for branch in eval_branches:
        metrics[f"ndcg_{branch}"] = 0.0
        metrics[f"coverage_{branch}"] = 0.0
    return metrics


@torch.no_grad()
def compute_eval_batch_stats(
    batch,
    encoder,
    compressor,
    node_emb,
    graph_meta,
    k,
    device,
    eval_branches=("big", "small"),
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
    branch_outputs = BranchOutputs(z_big=z_big, z_small=z_small)

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

    stats = {"n_users": int(users.numel())}
    k = min(k, graph_meta.num_items)

    for branch in eval_branches:
        z = branch_outputs.select(branch)
        logits = logits_from_embeddings(z, users, item_ids_global)
        topk = torch.topk(logits, k=k, dim=1).indices

        ndcg_sum = 0.0
        topk_union = set()

        for i in range(len(users)):
            pos = pos_items[i]
            hits = (topk[i] == pos).nonzero(as_tuple=False)
            if hits.numel() > 0:
                rank = hits[0].item()
                ndcg_sum += 1.0 / math.log2(rank + 2)
            topk_union.update(topk[i].detach().cpu().tolist())

        stats[f"ndcg_{branch}_sum"] = ndcg_sum
        stats[f"topk_union_{branch}"] = topk_union

    return stats


def aggregate_eval_stats(outputs, num_items: int, eval_branches=("big", "small")):
    if len(outputs) == 0:
        return _empty_metrics(eval_branches)

    total_users = sum(out["n_users"] for out in outputs)
    denom = max(total_users, 1)
    metrics = {}

    for branch in eval_branches:
        total_ndcg = 0.0
        topk_union = set()

        for out in outputs:
            total_ndcg += out.get(f"ndcg_{branch}_sum", 0.0)
            topk_union.update(out.get(f"topk_union_{branch}", set()))

        metrics[f"ndcg_{branch}"] = total_ndcg / denom
        metrics[f"coverage_{branch}"] = len(topk_union) / num_items

    return metrics
