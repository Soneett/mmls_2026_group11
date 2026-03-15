from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

@dataclass
class TrainState:

    encoder: nn.Module
    compressor: nn.Module
    node_emb: nn.Module

    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]

@dataclass
class GraphMeta:

    num_nodes: int
    num_items: int
    item_offset: int
