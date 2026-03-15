import torch
import numpy as np
from collections import deque

from ..graph.graph_compose import mp_edges_for_sid, compute_z_from_prefix

@torch.no_grad()
def eval_streaming(
    events_by_sid,
    mp_by_sid,
    start_prefix_sid,
    window_sids,
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

    window = deque(maxlen=(window_sids if window_sids > 0 else None))

    prefix_sid = start_prefix_sid

    ndcg_big = []
    ndcg_small = []

    topk_union_big = set()
    topk_union_small = set()

    for sid in sorted(events_by_sid.keys()):

        while prefix_sid < sid:

            s, d = mp_edges_for_sid(prefix_sid, mp_by_sid, device)

            if s.numel() > 0:
                window.append((s, d))

            prefix_sid += 1

        z_big, z_small = compute_z_from_prefix(
            window,
            graph_meta.num_nodes,
            state.encoder,
            state.compressor,
            state.node_emb,
            device,
        )

        batch = events_by_sid[sid]

        if len(batch) == 0:
            continue

        users = torch.tensor(
            batch["from"].values,
            device=device,
            dtype=torch.long,
        )

        pos_items = torch.tensor(
            batch["to"].values - graph_meta.item_offset,
            device=device,
            dtype=torch.long,
        )

        logits_big = z_big[users] @ z_big[item_ids_global].t()
        logits_small = z_small[users] @ z_small[item_ids_global].t()

        topk_big = torch.topk(logits_big, k=k, dim=1).indices
        topk_small = torch.topk(logits_small, k=k, dim=1).indices

        for i in range(len(users)):

            pos = pos_items[i]

            # big
            hits_big = (topk_big[i] == pos).nonzero(as_tuple=False)

            if hits_big.numel() == 0:
                ndcg_big.append(0)
            else:
                rank = hits_big[0].item()
                ndcg_big.append(1 / np.log2(rank + 2))

            topk_union_big.update(topk_big[i].cpu().tolist())

            # small
            hits_small = (topk_small[i] == pos).nonzero(as_tuple=False)

            if hits_small.numel() == 0:
                ndcg_small.append(0)
            else:
                rank = hits_small[0].item()
                ndcg_small.append(1 / np.log2(rank + 2))

            topk_union_small.update(topk_small[i].cpu().tolist())

    coverage_big = len(topk_union_big) / graph_meta.num_items
    coverage_small = len(topk_union_small) / graph_meta.num_items

    return {
        "ndcg_big": float(np.mean(ndcg_big)),
        "ndcg_small": float(np.mean(ndcg_small)),
        "coverage_big": coverage_big,
        "coverage_small": coverage_small,
    }