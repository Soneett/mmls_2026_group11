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

    distillation_mode: str
    distillation_weight: float
    lambda_kd: float
    kd_temperature: float
    lambda_emb: float
    teacher_checkpoint: str
    teacher_config: str
    grad_clip: float
    precision: str

    use_scheduler: bool
    use_grad_checkpointing: bool

    project: str
    run_name: str
    checkpoint_dir: str


def load_config(path: str) -> CFG:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    defaults = {
        "distillation_mode": "joint",
        "distillation_weight": 0.1,
        "lambda_kd": 0.0,
        "kd_temperature": 1.0,
        "lambda_emb": 0.0,
        "teacher_checkpoint": "",
        "teacher_config": "",
        "precision": "32-true",
    }
    for key, value in defaults.items():
        data.setdefault(key, value)
    return CFG(**data)
