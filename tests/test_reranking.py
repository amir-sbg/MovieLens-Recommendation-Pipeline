import numpy as np
import pytest

from recsys_lab.reranking import mmr_rerank


def test_mmr_moves_redundant_item_behind_diverse_candidate() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ]
    )
    scores = {0: 1.0, 1: 0.95, 2: 0.85}

    reranked = mmr_rerank([0, 1, 2], scores, embeddings, k=3, relevance_weight=0.5)

    assert reranked == [0, 2, 1]


def test_mmr_reduces_to_relevance_order_at_weight_one() -> None:
    embeddings = np.eye(3)

    reranked = mmr_rerank(
        [2, 0, 1],
        {0: 0.7, 1: 0.8, 2: 0.9},
        embeddings,
        relevance_weight=1.0,
    )

    assert reranked == [2, 1, 0]


def test_mmr_validates_embedding_indices() -> None:
    with pytest.raises(ValueError, match="outside"):
        mmr_rerank([3], {3: 1.0}, np.eye(3))
