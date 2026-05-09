from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.graph_compose import compute_z_from_edges
from src.training.runner import init_models


@dataclass(frozen=True)
class BranchOutputs:
    z_big: torch.Tensor
    z_small: torch.Tensor

    def select(self, branch: str) -> torch.Tensor:
        if branch == "big":
            return self.z_big
        if branch == "small":
            return self.z_small
        raise ValueError(f"Unknown branch '{branch}'. Expected 'big' or 'small'.")


class FrozenTeacherModel(nn.Module):
    """Standalone teacher used only for external knowledge distillation."""

    def __init__(self, cfg, num_nodes: int):
        super().__init__()
        self.node_emb, self.encoder, self.compressor = init_models(cfg, num_nodes)
        self.freeze()

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, edge_src, edge_dst, graph_meta, device) -> BranchOutputs:
        self.freeze()
        z_big, z_small = compute_z_from_edges(
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=graph_meta.num_nodes,
            encoder=self.encoder,
            compressor=self.compressor,
            node_emb=self.node_emb,
            device=device,
        )
        return BranchOutputs(z_big=z_big, z_small=z_small)


def _extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    module_keys = ("node_emb.", "encoder.", "compressor.")
    return {
        key: value
        for key, value in state_dict.items()
        if key.startswith(module_keys)
    }


def load_frozen_teacher(checkpoint_path: str, cfg, num_nodes: int, map_location="cpu") -> FrozenTeacherModel:
    teacher = FrozenTeacherModel(cfg=cfg, num_nodes=num_nodes)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_model_state_dict(checkpoint)

    if len(state_dict) == 0:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not contain node_emb/encoder/compressor weights."
        )

    teacher.load_state_dict(state_dict, strict=True)
    teacher.freeze()
    return teacher


def logits_from_embeddings(z: torch.Tensor, users: torch.Tensor, item_ids_global: torch.Tensor) -> torch.Tensor:
    return z[users] @ z[item_ids_global].t()


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("KD temperature must be positive.")

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
