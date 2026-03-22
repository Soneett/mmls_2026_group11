from dataclasses import dataclass
import yaml


@dataclass
class CFG:
    seed: int
    device: str
    epochs: int
    lr: float
    k: int
    n_layers: int
    embed_dim: int
    compressed_dim: int
    window_sids: int
    wandb: dict


def load_config(path: str) -> CFG:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return CFG(**data)