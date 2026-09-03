from __future__ import annotations

import pandas as pd
import pytest

from recsys_lab.data import (
    encode_splits,
    filter_min_interactions,
    generate_synthetic_ratings,
    prepare_recommendation_data,
    relevant_items_by_user,
    temporal_user_split,
    user_seen_items,
)


def test_synthetic_ratings_are_reproducible() -> None:
    first = generate_synthetic_ratings(users=5, items=20, density=0.2, seed=7)
    second = generate_synthetic_ratings(users=5, items=20, density=0.2, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert {"user_id", "item_id", "rating", "timestamp"}.issubset(first.columns)
    assert first["rating"].between(1, 5).all()


def test_temporal_split_keeps_last_two_events_per_user() -> None:
    interactions = generate_synthetic_ratings(users=4, items=10, density=0.4, seed=3)
    train, validation, test = temporal_user_split(interactions)

    assert len(validation) == 4
    assert len(test) == 4
    for user_id in interactions["user_id"].unique():
        source = interactions[interactions["user_id"] == user_id].sort_values("timestamp")
        assert test[test["user_id"] == user_id]["item_id"].iloc[0] == source["item_id"].iloc[-1]
        assert validation[validation["user_id"] == user_id]["item_id"].iloc[0] == source["item_id"].iloc[-2]


def test_encoding_adds_contiguous_user_and_item_indices() -> None:
    interactions = generate_synthetic_ratings(users=3, items=8, density=0.5, seed=4)
    encoded = encode_splits(*temporal_user_split(interactions))

    assert encoded.n_users == 3
    assert set(encoded.train["user_idx"]).issubset(set(range(encoded.n_users)))
    assert set(encoded.test["item_idx"]).issubset(set(range(encoded.n_items)))


def test_prepare_synthetic_data_returns_nonempty_splits() -> None:
    data = prepare_recommendation_data(users=10, items=30, density=0.2, seed=9)

    assert len(data.train) > len(data.validation)
    assert len(data.validation) == len(data.test) == data.n_users


def test_filter_min_interactions_rejects_empty_result() -> None:
    frame = pd.DataFrame(
        {
            "user_id": ["a", "b"],
            "item_id": ["i1", "i2"],
            "rating": [5, 4],
            "timestamp": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="no users remain"):
        filter_min_interactions(frame, min_interactions=3)


def test_seen_and_relevant_items_are_grouped_by_user() -> None:
    frame = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "item_idx": [2, 3, 2],
            "rating": [5.0, 2.0, 4.0],
        }
    )

    assert user_seen_items(frame) == {0: {2, 3}, 1: {2}}
    assert relevant_items_by_user(frame) == {0: {2}, 1: {2}}
