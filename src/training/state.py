from dataclasses import dataclass


@dataclass
class GraphMeta:
    num_nodes: int
    num_items: int
    item_offset: int