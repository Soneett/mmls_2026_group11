import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import wandb

from collections import deque

from ..graph.graph_compose import mp_edges_for_sid, compute_z_from_prefix
from ..dataset.preprocessing import select_last_event_per_user

def train_epoch_streaming(
    train_events_by_sid,
    mp_by_sid,
    window_sids,
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

    window = deque(maxlen=(window_sids if window_sids > 0 else None))

    prefix_sid = 0

    total_loss = 0.0
    total_users = 0

    for sid in sorted(train_events_by_sid.keys()):

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

        batch = train_events_by_sid[sid]

        if len(batch) == 0:
            continue

        users_cpu, pos_items_cpu = select_last_event_per_user(
            batch,
            graph_meta.item_offset
        )

        if users_cpu.numel() == 0:
            continue

        users = users_cpu.to(device)
        pos_items = pos_items_cpu.to(device)

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
            logits_big.detach()
        )

        loss = loss_big + loss_small + cfg.distillation_weight * distill_loss

        state.optimizer.zero_grad(set_to_none=True)

        loss.backward()

        nn_utils.clip_grad_norm_(
            list(state.encoder.parameters()) +
            list(state.compressor.parameters()),
            cfg.grad_clip,
        )

        state.optimizer.step()

        if state.scheduler is not None:
            state.scheduler.step()

        U = users.numel()

        total_loss += loss.item() * U
        total_users += U

    mean_loss = total_loss / max(total_users, 1)

    wandb.log({"train_loss": mean_loss})

    return mean_loss