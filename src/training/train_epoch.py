import torch
import torch.nn.functional as F
import torch.distributed as dist

from ..graph.graph_compose import compute_z_from_edges


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _zero_from_graph(z_big, z_small):
    # Нулевой loss, но завязан на граф вычислений всех веток
    return (z_big.sum() + z_small.sum()) * 0.0


def compute_train_batch_loss(
    batch,
    encoder,
    compressor,
    node_emb,
    graph_meta,
    distillation_weight,
    device,
):
    prefix_src = batch.prefix_src.to(device, non_blocking=True)
    prefix_dst = batch.prefix_dst.to(device, non_blocking=True)

    z_big, z_small = compute_z_from_edges(
        edge_src=prefix_src,
        edge_dst=prefix_dst,
        num_nodes=graph_meta.num_nodes,
        encoder=encoder,
        compressor=compressor,
        node_emb=node_emb,
        device=device,
    )

    batch_targets = batch.events
    local_user_count = int(len(batch_targets))

    world_size = dist.get_world_size() if _is_dist() else 1

    local_users = torch.tensor([float(local_user_count)], device=device)
    global_users = local_users.clone()
    if _is_dist():
        dist.all_reduce(global_users, op=dist.ReduceOp.SUM)
    global_users = torch.clamp(global_users, min=1.0)

    user_scale = world_size * local_users / global_users
    zero = _zero_from_graph(z_big, z_small)

    if local_user_count == 0:
        loss_big_local = zero
        loss_small_local = zero
        distill_local = zero
        local_logits_elems = torch.tensor([0.0], device=device)
    else:
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

        users_z_big = z_big[users]
        items_z_big = z_big[item_ids_global]
        logits_big = users_z_big @ items_z_big.t()
        loss_big_local = F.cross_entropy(logits_big, pos_items, reduction="mean")

        users_z_small = z_small[users]
        items_z_small = z_small[item_ids_global]
        logits_small = users_z_small @ items_z_small.t()
        loss_small_local = F.cross_entropy(logits_small, pos_items, reduction="mean")

        distill_local = F.mse_loss(logits_small, logits_big.detach(), reduction="mean")
        local_logits_elems = torch.tensor([float(logits_small.numel())], device=device)

    global_logits_elems = local_logits_elems.clone()
    if _is_dist():
        dist.all_reduce(global_logits_elems, op=dist.ReduceOp.SUM)
    global_logits_elems = torch.clamp(global_logits_elems, min=1.0)

    distill_scale = world_size * local_logits_elems / global_logits_elems

    loss_big = loss_big_local * user_scale
    loss_small = loss_small_local * user_scale
    distill_loss = distill_local * distill_scale

    loss = loss_big + loss_small + distillation_weight * distill_loss

    return {
        "loss": loss,
        "loss_big": loss_big_local.detach(),
        "loss_small": loss_small_local.detach(),
        "distill_loss": distill_local.detach(),
        "n_users": local_user_count,
    }
