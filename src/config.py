from dataclasses import dataclass
import yaml


@dataclass
class CFG:
    ml100k_path: str
    sep: str
    snapshot_gran: str
    window_sids: int

    node_dim: int
    embed_dim: int
    compressed_dim: int

    n_layers: int
    dropout: float

    lr: float
    weight_decay: float
    epochs: int

    train_ratio: float
    val_ratio: float
    test_ratio: float

    k: int

    seed: int
    device: str

    debug_sids: int
    full_train_epochs: int

    distillation_weight: float
    grad_clip: float

    use_scheduler: bool
    use_grad_checkpointing: bool
    parallel_mode: str
    devices: int
    grad_accum_steps: int
    users_per_batch: int

    project: str
    run_name: str
    checkpoint_dir: str


def load_config(path: str) -> CFG:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return CFG(**data)
