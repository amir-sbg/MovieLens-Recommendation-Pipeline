from __future__ import annotations

import numpy as np


def mmr_rerank(
    candidate_items: list[int],
    relevance_scores: dict[int, float],
    item_embeddings: np.ndarray,
    k: int = 10,
    relevance_weight: float = 0.8,
) -> list[int]:
    """Greedily trade relevance against similarity to already selected items."""
    if k < 1:
        raise ValueError("k must be positive")
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between 0 and 1")
    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must be a 2-D array")
    if len(set(candidate_items)) != len(candidate_items):
        raise ValueError("candidate_items must be unique")
    if any(item not in relevance_scores for item in candidate_items):
        raise ValueError("every candidate needs a relevance score")
    if any(item < 0 or item >= item_embeddings.shape[0] for item in candidate_items):
        raise ValueError("candidate item index is outside the embedding table")
    if not candidate_items:
        return []

    scores = np.asarray([relevance_scores[item] for item in candidate_items], dtype=float)
    span = float(scores.max() - scores.min())
    normalized_scores = (scores - scores.min()) / span if span > 0 else np.ones_like(scores)
    vectors = item_embeddings[candidate_items].astype(float, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    remaining = list(range(len(candidate_items)))
    selected: list[int] = []
    while remaining and len(selected) < min(k, len(candidate_items)):
        best_index = remaining[0]
        best_score = -np.inf
        for index in remaining:
            redundancy = (
                max(float(vectors[index] @ vectors[chosen]) for chosen in selected)
                if selected
                else 0.0
            )
            mmr = relevance_weight * normalized_scores[index] - (1.0 - relevance_weight) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_index = index
        selected.append(best_index)
        remaining.remove(best_index)
    return [candidate_items[index] for index in selected]
