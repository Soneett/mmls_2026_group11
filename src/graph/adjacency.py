import torch
from typing import List, Tuple

def concat_edges(edge_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(edge_list) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    src = torch.cat([s for s, _ in edge_list], dim=0)
    dst = torch.cat([d for _, d in edge_list], dim=0)
    return src, dst


def build_norm_adj(edge_src: torch.Tensor, edge_dst: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    self_idx = torch.arange(num_nodes, dtype=torch.long, device=device)
    src = torch.cat([edge_src, self_idx], dim=0)
    dst = torch.cat([edge_dst, self_idx], dim=0)

    deg = torch.bincount(src, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)

    vals = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    idx = torch.stack([src, dst], dim=0)

    A = torch.sparse_coo_tensor(idx, vals, size=(num_nodes, num_nodes), device=device)
    A = A.coalesce()
    return A