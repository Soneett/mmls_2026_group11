import torch
import torch.nn.functional as F

from ..graph.graph_compose import compute_z_from_edges
from ..dataset.preprocessing import select_last_event_per_user


def compute_train_batch_loss(
    batch,
    encoder,
    compressor,
    node_emb,
    graph_meta,
    distillation_mode,
    distillation_weight,
    lambda_kd,
    kd_temperature,
    teacher_outputs,
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

    users_z_big = z_big[users]
    items_z_big = z_big[item_ids_global]
    logits_big = users_z_big @ items_z_big.t()
    users_z_small = z_small[users]
    items_z_small = z_small[item_ids_global]
    logits_small = users_z_small @ items_z_small.t()
    loss_small = F.cross_entropy(logits_small, pos_items)

    if distillation_mode == "offline_kd":
        if teacher_outputs is None:
            raise ValueError("teacher_outputs must be provided for offline_kd mode")
        teacher_logits = teacher_outputs["logits"].detach().float()
        student_logits = logits_small.float()
        temperature = max(float(kd_temperature), 1e-8)
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        loss = loss_small + float(lambda_kd) * distill_loss
        loss_big = torch.zeros_like(loss_small)
    else:
        loss_big = F.cross_entropy(logits_big, pos_items)
        distill_loss = F.mse_loss(logits_small, logits_big.detach())
        loss = loss_big + loss_small + float(distillation_weight) * distill_loss

    return {
        "loss": loss,
        "loss_big": loss_big.detach(),
        "loss_small": loss_small.detach(),
        "distill_loss": distill_loss.detach(),
        "n_users": int(users.numel()),
    }
