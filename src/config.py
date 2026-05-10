from dataclasses import dataclass

import yaml


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

    lr: float = 0.0003
    weight_decay: float = 0.00001
    epochs: int = 50

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    k: int = 20

    seed: int = 1337
    device: str = "cpu"

    debug_sids: int = 3
    full_train_epochs: int = 30

    distillation_mode: str
    distillation_weight: float
    lambda_kd: float
    kd_temperature: float
    lambda_emb: float
    teacher_checkpoint: str
    teacher_config: str
    grad_clip: float
    precision: str

    use_scheduler: bool = False
    use_grad_checkpointing: bool = False
    parallel_mode: str = "none"
    devices: int = 1
    grad_accum_steps: int = 1
    users_per_batch: int = 0

    project: str = "dynamic_gnn_recommender_movielens"
    run_name: str = "gcn_L2_emb64_comp16"
    checkpoint_dir: str = "checkpoints"

    # distributed / deepspeed
    num_nodes: int = 1
    zero_stage: int = 1
    zero_offload_optimizer: bool = False
    zero_offload_optimizer_device: str = "cpu"


def load_config(path: str) -> CFG:
    with open(path, "r", encoding="utf-8") as f:
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
