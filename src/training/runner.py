import torch

from models.encoder import SimpleGCNEncoder
from models.compressor import Compressor


def init_models(cfg, num_nodes):
    node_emb = torch.nn.Embedding(num_nodes, cfg.node_dim)
    torch.nn.init.normal_(node_emb.weight, std=0.1)

    encoder = SimpleGCNEncoder(
        in_dim=cfg.node_dim,
        hid_dim=cfg.embed_dim,
        out_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )

    compressor = Compressor(
        cfg.embed_dim,
        cfg.compressed_dim,
    )

    return node_emb, encoder, compressor


def init_optimizer(cfg, node_emb, encoder, compressor):
    return torch.optim.Adam(
        list(node_emb.parameters())
        + list(encoder.parameters())
        + list(compressor.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def init_scheduler(cfg, opt):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.full_train_epochs,
    )