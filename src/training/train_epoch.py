import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import wandb

from ..graph.graph_compose import compute_z_from_edges
from ..dataset.preprocessing import select_last_event_per_user


def train_epoch_streaming(
    train_loader,
    state,
    graph_meta,
    cfg,
    device,
):
    state.encoder.train()
    state.compressor.train()
    state.node_emb.train()

    item_ids_global = torch.arange(
        graph_meta.item_offset,
        graph_meta.item_offset + graph_meta.num_items,
        device=device,
        dtype=torch.long,
    )

    total_loss = 0.0
    total_users = 0

    for batch in train_loader:
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

        users_z_big = z_big[users]
        items_z_big = z_big[item_ids_global]
        logits_big = users_z_big @ items_z_big.t()
        loss_big = F.cross_entropy(logits_big, pos_items)

        users_z_small = z_small[users]
        items_z_small = z_small[item_ids_global]
        logits_small = users_z_small @ items_z_small.t()
        loss_small = F.cross_entropy(logits_small, pos_items)

        distill_loss = F.mse_loss(
            logits_small,
            logits_big.detach(),
        )

        loss = loss_big + loss_small + cfg.distillation_weight * distill_loss

        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        nn_utils.clip_grad_norm_(
            list(state.node_emb.parameters())
            + list(state.encoder.parameters())
            + list(state.compressor.parameters()),
            cfg.grad_clip,
        )

        state.optimizer.step()

        if state.scheduler is not None:
            state.scheduler.step()

        n_users = users.numel()
        total_loss += loss.item() * n_users
        total_users += n_users

    mean_loss = total_loss / max(total_users, 1)

    wandb.log({"train_loss": mean_loss})

    return mean_loss