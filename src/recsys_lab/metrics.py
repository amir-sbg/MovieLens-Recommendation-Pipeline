from __future__ import annotations

from itertools import combinations
from math import log2
from typing import Iterable

import numpy as np


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("cannot compute RMSE on an empty input")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        raise ValueError("cannot compute MAE on an empty input")
    return float(np.mean(np.abs(true - pred)))


def rating_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    return {
        "rmse": rmse(true, pred),
        "mae": mae(true, pred),
    }


def _top_k(items: list[int], k: int) -> list[int]:
    if k < 1:
        raise ValueError("k must be positive")
    return items[:k]


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top = _top_k(recommended, k)
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = _top_k(recommended, k)
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = _top_k(recommended, k)
    running_hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(top, start=1):
        if item in relevant:
            running_hits += 1
            precision_sum += running_hits / rank
    return precision_sum / min(len(relevant), k)


def reciprocal_rank_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    for rank, item in enumerate(_top_k(recommended, k), start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = _top_k(recommended, k)
    dcg = 0.0
    for rank, item in enumerate(top, start=1):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def ranking_metrics(
    recommendations_by_user: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    k: int = 10,
) -> dict[str, float]:
    users = [user for user, relevant in relevant_by_user.items() if relevant]
    if not users:
        return {
            "users": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "hit_rate": 0.0,
        }

    precisions = []
    recalls = []
    average_precisions = []
    reciprocal_ranks = []
    ndcgs = []
    hits = []
    for user in users:
        recommended = recommendations_by_user.get(user, [])
        relevant = relevant_by_user[user]
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        average_precisions.append(average_precision_at_k(recommended, relevant, k))
        reciprocal_ranks.append(reciprocal_rank_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        hits.append(float(any(item in relevant for item in _top_k(recommended, k))))

    return {
        "users": float(len(users)),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "map_at_k": float(np.mean(average_precisions)),
        "mrr_at_k": float(np.mean(reciprocal_ranks)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "hit_rate": float(np.mean(hits)),
    }


def catalog_coverage(recommendations_by_user: dict[int, list[int]], n_items: int) -> float:
    if n_items < 1:
        raise ValueError("n_items must be positive")
    recommended_items = {
        int(item)
        for recommendations in recommendations_by_user.values()
        for item in recommendations
    }
    return len(recommended_items) / n_items


def personalization(recommendations_by_user: dict[int, list[int]]) -> float:
    lists = [set(items) for items in recommendations_by_user.values() if items]
    if len(lists) < 2:
        return 0.0

    similarities = []
    for left, right in combinations(lists, 2):
        union = left | right
        if union:
            similarities.append(len(left & right) / len(union))
    if not similarities:
        return 0.0
    return float(1.0 - np.mean(similarities))


def novelty_at_k(
    recommendations_by_user: dict[int, list[int]],
    item_popularity: dict[int, int],
    total_interactions: int,
    k: int = 10,
) -> float:
    if total_interactions < 1:
        raise ValueError("total_interactions must be positive")
    values = []
    for recommendations in recommendations_by_user.values():
        for item in _top_k(recommendations, k):
            probability = item_popularity.get(int(item), 0) / total_interactions
            values.append(-float(np.log2(max(probability, 1.0 / total_interactions))))
    return float(np.mean(values)) if values else 0.0


def long_tail_share_at_k(
    recommendations_by_user: dict[int, list[int]],
    item_popularity: dict[int, int],
    quantile: float = 0.50,
    k: int = 10,
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if not item_popularity:
        return 0.0
    cutoff = float(np.quantile(list(item_popularity.values()), quantile))
    items = [
        int(item)
        for recommendations in recommendations_by_user.values()
        for item in _top_k(recommendations, k)
    ]
    if not items:
        return 0.0
    tail_items = [item for item in items if item_popularity.get(item, 0) <= cutoff]
    return len(tail_items) / len(items)
