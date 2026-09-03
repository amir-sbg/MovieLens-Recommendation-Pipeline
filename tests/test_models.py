from __future__ import annotations

import numpy as np

from recsys_lab.data import prepare_recommendation_data, user_seen_items
from recsys_lab.models import (
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


def test_matrix_factorization_recommendations_exclude_seen() -> None:
    data = prepare_recommendation_data(users=10, items=25, density=0.3, seed=14)
    seen = user_seen_items(data.train)
    model = MatrixFactorizationRecommender(factors=6, epochs=2, seed=14)
    model.fit(data.train, n_users=data.n_users, n_items=data.n_items)

    recommendations = model.recommend(0, seen_items=seen[0], k=5)

    assert len(recommendations) == 5
    assert set(recommendations).isdisjoint(seen[0])
