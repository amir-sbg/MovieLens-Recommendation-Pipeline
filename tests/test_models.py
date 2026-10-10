from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys_lab.data import prepare_recommendation_data, user_seen_items
from recsys_lab.models import (
    BPRMatrixFactorizationRecommender,
    ItemKNNRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender,
)


def test_popularity_recommender_excludes_seen_items() -> None:
    data = prepare_recommendation_data(users=8, items=20, density=0.3, seed=11)
    seen = user_seen_items(data.train)
    model = PopularityRecommender().fit(data.train, n_users=data.n_users, n_items=data.n_items)

    recommendations = model.recommend(0, seen_items=seen[0], k=5)

    assert len(recommendations) == 5
    assert set(recommendations).isdisjoint(seen[0])


def test_popularity_recommender_shrinks_sparse_item_means() -> None:
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 3],
            "item_idx": [0, 1, 1, 1],
            "rating": [5.0, 3.0, 4.0, 4.0],
        }
    )

    no_prior = PopularityRecommender(prior_weight=0.0).fit(interactions, n_items=2)
    smoothed = PopularityRecommender(prior_weight=10.0).fit(interactions, n_items=2)

    assert no_prior.item_means[0] == 5.0
    assert smoothed.item_means[0] < 5.0
    assert smoothed.item_means[0] > smoothed.global_mean


def test_popularity_recommender_rejects_negative_prior() -> None:
    data = prepare_recommendation_data(users=8, items=20, density=0.3, seed=11)

    with pytest.raises(ValueError, match="prior_weight"):
        PopularityRecommender(prior_weight=-1.0).fit(data.train)


def test_item_knn_scores_and_recommends() -> None:
    data = prepare_recommendation_data(users=10, items=25, density=0.3, seed=12)
    seen = user_seen_items(data.train)
    model = ItemKNNRecommender(n_neighbors=5).fit(
        data.train,
        n_users=data.n_users,
        n_items=data.n_items,
    )

    scores = model.scores_for_user(0)
    recommendations = model.recommend(0, seen_items=seen[0], k=5)

    assert scores.shape == (data.n_items,)
    assert np.isfinite(scores).all()
    assert set(recommendations).isdisjoint(seen[0])


def test_matrix_factorization_trains_and_predicts() -> None:
    data = prepare_recommendation_data(users=12, items=30, density=0.25, seed=13)
    model = MatrixFactorizationRecommender(factors=8, epochs=4, learning_rate=0.04, seed=13)

    model.fit(data.train, validation=data.validation, n_users=data.n_users, n_items=data.n_items)
    predictions = model.predict_pairs(
        data.test["user_idx"].to_numpy(dtype=int),
        data.test["item_idx"].to_numpy(dtype=int),
    )

    assert len(model.history) == 4
    assert np.isfinite(predictions).all()
    assert predictions.min() >= 1.0
    assert predictions.max() <= 5.0


def test_matrix_factorization_falls_back_for_unknown_ids() -> None:
    data = prepare_recommendation_data(users=10, items=14, density=0.35, seed=4)
    model = MatrixFactorizationRecommender(factors=4, epochs=2, seed=4).fit(
        data.train,
        validation=data.validation,
        n_users=data.n_users,
        n_items=data.n_items,
    )

    predictions = model.predict_pairs(
        np.array([data.n_users + 1, 0]),
        np.array([0, data.n_items + 1]),
    )
    cold_scores = model.scores_for_user(data.n_users + 1)

    assert predictions.shape == (2,)
    assert np.all((predictions >= 1.0) & (predictions <= 5.0))
    assert cold_scores.shape == (data.n_items,)


def test_matrix_factorization_early_stops_on_flat_validation() -> None:
    data = prepare_recommendation_data(users=12, items=30, density=0.25, seed=19)
    model = MatrixFactorizationRecommender(
        factors=6,
        epochs=8,
        learning_rate=0.0,
        patience=1,
        min_delta=1e-8,
        seed=19,
    )

    model.fit(data.train, validation=data.validation, n_users=data.n_users, n_items=data.n_items)

    assert len(model.history) == 2
    assert "validation_rmse" in model.history[0]


def test_matrix_factorization_rejects_bad_training_settings() -> None:
    data = prepare_recommendation_data(users=8, items=20, density=0.3, seed=18)

    with pytest.raises(ValueError, match="patience"):
        MatrixFactorizationRecommender(patience=0).fit(data.train)
    with pytest.raises(ValueError, match="learning_rate"):
        MatrixFactorizationRecommender(learning_rate=-0.1).fit(data.train)


def test_matrix_factorization_recommendations_exclude_seen() -> None:
    data = prepare_recommendation_data(users=10, items=25, density=0.3, seed=14)
    seen = user_seen_items(data.train)
    model = MatrixFactorizationRecommender(factors=6, epochs=2, seed=14)
    model.fit(data.train, n_users=data.n_users, n_items=data.n_items)

    recommendations = model.recommend(0, seen_items=seen[0], k=5)

    assert len(recommendations) == 5
    assert set(recommendations).isdisjoint(seen[0])


def test_matrix_factorization_artifact_round_trip(tmp_path) -> None:
    data = prepare_recommendation_data(users=8, items=18, density=0.35, seed=22)
    model = MatrixFactorizationRecommender(factors=4, epochs=2, seed=22).fit(
        data.train,
        n_users=data.n_users,
        n_items=data.n_items,
    )
    path = tmp_path / "mf.npz"
    model.save_npz(path)
    restored = MatrixFactorizationRecommender.load_npz(path)

    np.testing.assert_allclose(
        model.predict_pairs(
            data.test["user_idx"].to_numpy(),
            data.test["item_idx"].to_numpy(),
        ),
        restored.predict_pairs(
            data.test["user_idx"].to_numpy(),
            data.test["item_idx"].to_numpy(),
        ),
    )


def test_bpr_learns_pairwise_rankings_and_excludes_seen_items(tmp_path) -> None:
    data = prepare_recommendation_data(users=12, items=30, density=0.3, seed=31)
    seen = user_seen_items(data.train)
    model = BPRMatrixFactorizationRecommender(
        factors=6,
        epochs=3,
        samples_per_epoch=80,
        seed=31,
    ).fit(data.train, n_users=data.n_users, n_items=data.n_items)

    recommendations = model.recommend(0, seen_items=seen[0], k=5)
    model.save_npz(tmp_path / "bpr.npz")

    assert len(model.history) == 3
    assert all(np.isfinite(row["pairwise_loss"]) for row in model.history)
    assert len(recommendations) == 5
    assert set(recommendations).isdisjoint(seen[0])
    assert (tmp_path / "bpr.npz").exists()
