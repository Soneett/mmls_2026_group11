import torch
from collections import deque
from typing import Dict, Tuple

from .adjacency import concat_edges, build_norm_adj

def mp_edges_for_sid(sid, mp_by_sid, device):

    g = mp_by_sid.get(int(sid))

    if g is None or len(g) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    src = torch.tensor(g["from"].to_numpy(), dtype=torch.long, device=device)
    dst = torch.tensor(g["to"].to_numpy(), dtype=torch.long, device=device)

    return src, dst


def compute_z_from_prefix(window: deque, num_nodes, encoder, compressor, node_emb, device):

    src, dst = concat_edges(list(window))

    A = build_norm_adj(src, dst, num_nodes=num_nodes, device=device)

    z_big = encoder(A, node_emb.weight)

    z_small = compressor(z_big)

    return z_big, z_small
