import torch
import torch.nn.functional as F

from ..graph.graph_compose import compute_z_from_edges
from ..dataset.preprocessing import select_last_event_per_user
from .distillation import (
    BranchOutputs,
    distillation_kl_loss,
    logits_from_embeddings,
)


def compute_train_batch_loss(
    batch,
    encoder,
    compressor,
    node_emb,
    graph_meta,
    distillation_weight,
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
    loss_big = F.cross_entropy(logits_big, pos_items)

    users_z_small = z_small[users]
    items_z_small = z_small[item_ids_global]
    logits_small = users_z_small @ items_z_small.t()
    loss_small = F.cross_entropy(logits_small, pos_items)

    distill_loss = F.mse_loss(logits_small, logits_big.detach())
    loss = loss_big + loss_small + distillation_weight * distill_loss

    return {
        "loss": loss,
        "loss_big": loss_big.detach(),
        "loss_small": loss_small.detach(),
        "distill_loss": distill_loss.detach(),
        "n_users": int(users.numel()),
    }


def compute_student_ce_batch_loss(
    batch,
    encoder,
    compressor,
    node_emb,
    graph_meta,
    student_branch,
    device,
):
    """Train a single student branch with recommendation cross-entropy only."""
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
    student_outputs = BranchOutputs(z_big=z_big, z_small=z_small)

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

    student_z = student_outputs.select(student_branch)
    student_logits = logits_from_embeddings(student_z, users, item_ids_global)
    loss_ce = F.cross_entropy(student_logits, pos_items)

    return {
        "loss": loss_ce,
        "loss_ce": loss_ce.detach(),
        "n_users": int(users.numel()),
    }


def compute_external_distillation_batch_loss(
    batch,
    encoder,
    compressor,
    node_emb,
    teacher,
    graph_meta,
    kd_weight,
    kd_temperature,
    teacher_branch,
    student_branch,
    device,
):
    """Train one student branch against labels and a frozen teacher checkpoint."""
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
    student_outputs = BranchOutputs(z_big=z_big, z_small=z_small)

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

    student_z = student_outputs.select(student_branch)
    student_logits = logits_from_embeddings(student_z, users, item_ids_global)
    loss_ce = F.cross_entropy(student_logits, pos_items)

    with torch.no_grad():
        teacher_outputs = teacher(
            edge_src=batch.prefix_src,
            edge_dst=batch.prefix_dst,
            graph_meta=graph_meta,
            device=device,
        )
        teacher_z = teacher_outputs.select(teacher_branch)
        teacher_logits = logits_from_embeddings(teacher_z, users, item_ids_global)

    loss_kd = distillation_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=kd_temperature,
    )
    loss = loss_ce + kd_weight * loss_kd

    return {
        "loss": loss,
        "loss_ce": loss_ce.detach(),
        "loss_kd": loss_kd.detach(),
        "n_users": int(users.numel()),
    }
