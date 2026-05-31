import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.dataset.temporal_dataset import build_temporal_graph_dataset
from src.graph.graph_compose import concat_edges, compute_z_from_edges
from src.training.runner import init_models
from src.utils.seed import seed_everything


def _load_teacher_weights(cfg, num_nodes: int):
    if not getattr(cfg, "teacher_checkpoint", ""):
        raise ValueError("teacher_checkpoint must be set")
    if not getattr(cfg, "teacher_config", ""):
        raise ValueError("teacher_config must be set")

    teacher_cfg = load_config(cfg.teacher_config)
    node_emb, encoder, compressor = init_models(teacher_cfg, num_nodes)

    ckpt = torch.load(cfg.teacher_checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    node_emb.load_state_dict({k.replace("node_emb.", "", 1): v for k, v in state.items() if k.startswith("node_emb.")})
    encoder.load_state_dict({k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")})
    compressor.load_state_dict({k.replace("compressor.", "", 1): v for k, v in state.items() if k.startswith("compressor.")})

    node_emb.eval()
    encoder.eval()
    compressor.eval()
    return node_emb, encoder, compressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="artifacts/teacher_embeddings.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    dataset = build_temporal_graph_dataset(cfg)
    node_emb, encoder, compressor = _load_teacher_weights(cfg, dataset.num_nodes)

    edge_list = []
    for sid in sorted(dataset.mp_by_sid.keys()):
        df = dataset.mp_by_sid[sid]
        src = torch.tensor(df["from"].to_numpy(), dtype=torch.long)
        dst = torch.tensor(df["to"].to_numpy(), dtype=torch.long)
        edge_list.append((src, dst))

    edge_src, edge_dst = concat_edges(edge_list)

    with torch.no_grad():
        z_big, z_small = compute_z_from_edges(
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=dataset.num_nodes,
            encoder=encoder,
            compressor=compressor,
            node_emb=node_emb,
            device=torch.device("cpu"),
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z_big": z_big.cpu(), "z_small": z_small.cpu()}, output_path)
    print(f"Saved teacher embeddings to {output_path}")


if __name__ == "__main__":
    main()
