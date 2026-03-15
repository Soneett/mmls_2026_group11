from dataclasses import dataclass

@dataclass
class CFG:
    ml100k_path: str = "data/ml100k_ratings.csv"
    sep: str = ";"

    snapshot_gran: str = "d"
    window_sids: int = 0

    node_dim: int = 64
    embed_dim: int = 64
    compressed_dim: int = 16

    n_layers: int = 2
    dropout: float = 0.1

    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 50

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    k: int = 20

    seed: int = 1337
    device: str = "cpu"

    debug_sids: int = 3
    full_train_epochs: int = 30

    distillation_weight: float = 0.1
    grad_clip: float = 1.0
