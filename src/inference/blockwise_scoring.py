from __future__ import annotations

import torch


def blockwise_topk_dot_product(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    k: int,
    item_block_size: int = 1024,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k dot-product scores without materializing all scores.

    This function is equivalent to:

        scores = user_embeddings @ item_embeddings.T
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)

    but it processes item embeddings by blocks. This reduces peak memory from
    O(num_users * num_items) to roughly O(num_users * (k + item_block_size)).

    Args:
        user_embeddings: Tensor with shape [num_users, dim].
        item_embeddings: Tensor with shape [num_items, dim].
        k: Number of items to return per user.
        item_block_size: Number of item embeddings scored in one block.
        largest: Same meaning as in torch.topk. For recommendations this should
            usually remain True.
        sorted: Same meaning as in torch.topk.

    Returns:
        A pair (topk_scores, topk_indices), both with shape [num_users, k].
        topk_indices are row indices into item_embeddings.
    """
    if user_embeddings.ndim != 2:
        raise ValueError(
            f"user_embeddings must be 2D, got shape {tuple(user_embeddings.shape)}"
        )

    if item_embeddings.ndim != 2:
        raise ValueError(
            f"item_embeddings must be 2D, got shape {tuple(item_embeddings.shape)}"
        )

    if user_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ValueError(
            "user_embeddings and item_embeddings must have the same embedding dim, "
            f"got {user_embeddings.shape[1]} and {item_embeddings.shape[1]}"
        )

    num_users = user_embeddings.shape[0]
    num_items = item_embeddings.shape[0]

    if num_users == 0:
        raise ValueError("user_embeddings must contain at least one user")

    if num_items == 0:
        raise ValueError("item_embeddings must contain at least one item")

    if not 1 <= k <= num_items:
        raise ValueError(f"k must be in [1, num_items], got k={k}, num_items={num_items}")

    if item_block_size <= 0:
        raise ValueError(f"item_block_size must be positive, got {item_block_size}")

    topk_scores = None
    topk_indices = None

    for start in range(0, num_items, item_block_size):
        end = min(start + item_block_size, num_items)
        item_block = item_embeddings[start:end]

        block_scores = user_embeddings @ item_block.t()
        block_indices = torch.arange(
            start,
            end,
            device=user_embeddings.device,
            dtype=torch.long,
        ).expand(num_users, end - start)

        if topk_scores is None:
            candidate_scores = block_scores
            candidate_indices = block_indices
        else:
            candidate_scores = torch.cat([topk_scores, block_scores], dim=1)
            candidate_indices = torch.cat([topk_indices, block_indices], dim=1)

        take_k = min(k, candidate_scores.shape[1])
        topk_scores, topk_pos = torch.topk(
            candidate_scores,
            k=take_k,
            dim=1,
            largest=largest,
            sorted=sorted,
        )
        topk_indices = torch.gather(candidate_indices, dim=1, index=topk_pos)

    return topk_scores, topk_indices
