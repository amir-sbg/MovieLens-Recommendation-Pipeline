from __future__ import annotations

import pytest

from recsys_lab.metrics import (
    average_precision_at_k,
    catalog_coverage,
    long_tail_share_at_k,
    ndcg_at_k,
    novelty_at_k,
    personalization,
    precision_at_k,
    ranking_metrics,
    recall_at_k,
    rating_metrics,
    rmse,
)


def test_rating_metrics_are_computed() -> None:
    metrics = rating_metrics([5, 4, 1], [4, 4, 2])

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(rmse([5, 4, 1], [4, 4, 2]))


def test_rating_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        rmse([1, 2], [1])


def test_top_k_ranking_metrics() -> None:
    recommended = [9, 3, 2, 8]
    relevant = {2, 3}

    assert precision_at_k(recommended, relevant, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1.0)
    assert average_precision_at_k(recommended, relevant, k=3) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert ndcg_at_k(recommended, relevant, k=3) > 0.0


def test_ranking_metrics_average_across_users() -> None:
    recs = {0: [1, 2, 3], 1: [4, 5, 6]}
    relevant = {0: {2}, 1: {9}}

    metrics = ranking_metrics(recs, relevant, k=2)

    assert metrics["users"] == 2
    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert metrics["recall_at_k"] == pytest.approx(0.5)


def test_coverage_and_personalization() -> None:
    recs = {0: [1, 2], 1: [2, 3], 2: [4, 5]}

    assert catalog_coverage(recs, n_items=10) == pytest.approx(0.5)
    assert personalization(recs) > 0.0


def test_novelty_is_higher_for_less_popular_items() -> None:
    popular = {0: [1], 1: [1]}
    long_tail = {0: [9], 1: [9]}
    popularity = {1: 50, 9: 2}

    assert novelty_at_k(long_tail, popularity, total_interactions=100) > novelty_at_k(
        popular,
        popularity,
        total_interactions=100,
    )


def test_long_tail_share_tracks_less_common_recommendations() -> None:
    recs = {0: [1, 2], 1: [3, 4]}
    popularity = {1: 100, 2: 5, 3: 3, 4: 80}

    assert long_tail_share_at_k(recs, popularity, quantile=0.5, k=2) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="quantile"):
        long_tail_share_at_k(recs, popularity, quantile=1.0)
