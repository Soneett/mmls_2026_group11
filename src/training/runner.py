import torch

from models.encoder import SimpleGCNEncoder
from models.compressor import Compressor
from .state import TrainState


def init_models_and_opt(cfg, num_nodes, device):
    node_emb = torch.nn.Embedding(num_nodes, cfg.node_dim).to(device)
    torch.nn.init.normal_(node_emb.weight, std=0.1)

    encoder = SimpleGCNEncoder(
        in_dim=cfg.node_dim,
        hid_dim=cfg.embed_dim,
        out_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)

    compressor = Compressor(
        cfg.embed_dim,
        cfg.compressed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(node_emb.parameters())
        + list(encoder.parameters())
        + list(compressor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = None

    state = TrainState(
        encoder=encoder,
        compressor=compressor,
        node_emb=node_emb,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    return state


def init_scheduler(cfg, opt):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.full_train_epochs,
    )