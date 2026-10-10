from __future__ import annotations

import pytest

from recsys_lab.metrics import (
    average_precision_at_k,
    bootstrap_ranking_intervals,
    catalog_coverage,
    exposure_gini,
    long_tail_share_at_k,
    ndcg_at_k,
    novelty_at_k,
    normalized_exposure_entropy,
    paired_ranking_bootstrap,
    personalization,
    precision_at_k,
    ranking_metrics,
    recall_by_popularity_segment,
    reciprocal_rank_at_k,
    recall_at_k,
    rating_metrics,
    rmse,
)


def test_rating_metrics_are_computed() -> None:
    metrics = rating_metrics([5, 4, 1], [4, 4, 2])

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(rmse([5, 4, 1], [4, 4, 2]))


def test_rating_metrics_accept_one_pass_iterables() -> None:
    metrics = rating_metrics((value for value in [5, 4, 1]), (value for value in [4, 4, 2]))

    assert metrics["mae"] == pytest.approx(2 / 3)


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
    assert reciprocal_rank_at_k(recommended, relevant, k=3) == pytest.approx(0.5)


def test_ranking_metrics_do_not_count_duplicate_recommendations_twice() -> None:
    recommended = [3, 3, 8, 2]
    relevant = {2, 3}

    assert precision_at_k(recommended, relevant, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1.0)
    assert average_precision_at_k(recommended, relevant, k=3) <= 1.0


def test_ranking_metrics_average_across_users() -> None:
    recs = {0: [1, 2, 3], 1: [4, 5, 6]}
    relevant = {0: {2}, 1: {9}}

    metrics = ranking_metrics(recs, relevant, k=2)

    assert metrics["users"] == 2
    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["mrr_at_k"] == pytest.approx(0.25)


def test_bootstrap_ranking_intervals_are_reproducible() -> None:
    recs = {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8]}
    relevant = {0: {1}, 1: {9}, 2: {5}, 3: {10}}

    first = bootstrap_ranking_intervals(recs, relevant, k=2, n_resamples=200, seed=9)
    second = bootstrap_ranking_intervals(recs, relevant, k=2, n_resamples=200, seed=9)

    assert first == second
    assert first["recall_at_k_ci_low"] <= 0.5 <= first["recall_at_k_ci_high"]
    assert first["ndcg_at_k_ci_low"] <= first["ndcg_at_k_ci_high"]


def test_bootstrap_ranking_intervals_validate_settings() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ranking_intervals({}, {}, n_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ranking_intervals({}, {}, confidence=1.0)


def test_paired_ranking_bootstrap_detects_better_candidate() -> None:
    relevant = {user: {user + 10} for user in range(8)}
    baseline = {user: [99, 98] for user in relevant}
    candidate = {user: [user + 10, 99] for user in relevant}

    comparison = paired_ranking_bootstrap(
        baseline,
        candidate,
        relevant,
        k=2,
        n_resamples=200,
        seed=7,
    )

    assert comparison["recall_delta"] == pytest.approx(1.0)
    assert comparison["ndcg_delta"] == pytest.approx(1.0)
    assert comparison["probability_recall_improves"] == pytest.approx(1.0)


def test_coverage_and_personalization() -> None:
    recs = {0: [1, 2], 1: [2, 3], 2: [4, 5]}

    assert catalog_coverage(recs, n_items=10) == pytest.approx(0.5)
    assert personalization(recs) > 0.0


def test_exposure_metrics_detect_catalog_concentration() -> None:
    concentrated = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    spread = {0: [0, 1], 1: [2, 3], 2: [4, 5]}

    assert exposure_gini(concentrated, n_items=6, k=2) > exposure_gini(spread, n_items=6, k=2)
    assert normalized_exposure_entropy(concentrated, n_items=6, k=2) < normalized_exposure_entropy(
        spread, n_items=6, k=2
    )


def test_exposure_metrics_validate_catalog_indices() -> None:
    with pytest.raises(ValueError, match="outside the catalog"):
        exposure_gini({0: [5]}, n_items=5)


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


def test_segment_recall_separates_head_and_tail_hits() -> None:
    recs = {0: [1, 4], 1: [2, 5]}
    relevant = {0: {1, 5}, 1: {2, 4}}
    popularity = {1: 100, 2: 80, 4: 4, 5: 2}

    metrics = recall_by_popularity_segment(recs, relevant, popularity, quantile=0.5, k=2)

    assert metrics["head_recall_at_k"] == 1.0
    assert metrics["tail_recall_at_k"] == 0.0
    assert metrics["head_relevant_items"] == 2
    assert metrics["tail_relevant_items"] == 2
