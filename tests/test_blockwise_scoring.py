import pytest
import torch

from src.inference.blockwise_scoring import blockwise_topk_dot_product


def _full_topk(user_embeddings, item_embeddings, k, largest=True, sorted=True):
    scores = user_embeddings @ item_embeddings.t()
    return torch.topk(scores, k=k, dim=1, largest=largest, sorted=sorted)


def test_blockwise_topk_matches_full_matmul():
    torch.manual_seed(0)

    users = torch.randn(7, 5)
    items = torch.randn(23, 5)
    k = 4

    expected = _full_topk(users, items, k=k)
    actual_scores, actual_indices = blockwise_topk_dot_product(
        users,
        items,
        k=k,
        item_block_size=6,
    )

    assert torch.allclose(actual_scores, expected.values, atol=1e-6)
    assert torch.equal(actual_indices, expected.indices)


def test_blockwise_topk_works_when_k_is_larger_than_block_size():
    torch.manual_seed(1)

    users = torch.randn(3, 4)
    items = torch.randn(17, 4)
    k = 8

    expected = _full_topk(users, items, k=k)
    actual_scores, actual_indices = blockwise_topk_dot_product(
        users,
        items,
        k=k,
        item_block_size=3,
    )

    assert torch.allclose(actual_scores, expected.values, atol=1e-6)
    assert torch.equal(actual_indices, expected.indices)


def test_blockwise_topk_supports_smallest_scores():
    torch.manual_seed(2)

    users = torch.randn(5, 6)
    items = torch.randn(19, 6)
    k = 5

    expected = _full_topk(users, items, k=k, largest=False)
    actual_scores, actual_indices = blockwise_topk_dot_product(
        users,
        items,
        k=k,
        item_block_size=4,
        largest=False,
    )

    assert torch.allclose(actual_scores, expected.values, atol=1e-6)
    assert torch.equal(actual_indices, expected.indices)


def test_blockwise_topk_preserves_dtype_and_shape():
    torch.manual_seed(3)

    users = torch.randn(2, 3, dtype=torch.float64)
    items = torch.randn(11, 3, dtype=torch.float64)

    scores, indices = blockwise_topk_dot_product(users, items, k=6, item_block_size=2)

    assert scores.shape == (2, 6)
    assert indices.shape == (2, 6)
    assert scores.dtype == torch.float64
    assert indices.dtype == torch.long


@pytest.mark.parametrize(
    "users, items, k, block_size",
    [
        (torch.randn(2, 3, 4), torch.randn(5, 4), 2, 4),
        (torch.randn(2, 4), torch.randn(5, 3), 2, 4),
        (torch.randn(2, 4), torch.randn(5, 4), 0, 4),
        (torch.randn(2, 4), torch.randn(5, 4), 6, 4),
        (torch.randn(2, 4), torch.randn(5, 4), 2, 0),
    ],
)
def test_blockwise_topk_validates_inputs(users, items, k, block_size):
    with pytest.raises(ValueError):
        blockwise_topk_dot_product(users, items, k=k, item_block_size=block_size)
