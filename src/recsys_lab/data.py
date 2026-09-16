from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
REQUIRED_COLUMNS = ("user_id", "item_id", "rating", "timestamp")


@dataclass(frozen=True)
class RecommendationData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    user_mapping: dict[str, int]
    item_mapping: dict[str, int]
    item_titles: dict[int, str]


def generate_synthetic_ratings(
    users: int = 100,
    items: int = 150,
    density: float = 0.08,
    latent_dim: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    if users < 2 or items < 6:
        raise ValueError("synthetic data needs at least 2 users and 6 items")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    if latent_dim < 1:
        raise ValueError("latent_dim must be positive")

    rng = np.random.default_rng(seed)
    user_factors = rng.normal(0, 0.8, size=(users, latent_dim))
    item_factors = rng.normal(0, 0.8, size=(items, latent_dim))
    user_bias = rng.normal(0, 0.25, size=users)
    item_bias = rng.normal(0, 0.25, size=items)
    ratings_per_user = max(3, min(items, int(round(items * density))))

    rows = []
    timestamp = 1_700_000_000
    for user_idx in range(users):
        chosen_items = rng.choice(items, size=ratings_per_user, replace=False)
        for item_idx in chosen_items:
            signal = user_factors[user_idx] @ item_factors[item_idx] / np.sqrt(latent_dim)
            noisy_score = 3.3 + signal + user_bias[user_idx] + item_bias[item_idx]
            noisy_score += rng.normal(0, 0.35)
            rating = int(np.clip(np.rint(noisy_score), 1, 5))
            rows.append(
                {
                    "user_id": f"user_{user_idx:04d}",
                    "item_id": f"item_{item_idx:04d}",
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
            timestamp += int(rng.integers(1, 60))
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def load_movielens_100k(data_dir: Path = Path("data")) -> pd.DataFrame:
    root = data_dir / "ml-100k"
    ratings_path = root / "u.data"
    if not ratings_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir / "ml-100k.zip"
        urlretrieve(MOVIELENS_100K_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": str, "item_id": str, "rating": float, "timestamp": int},
    )
    return normalize_interactions(ratings)


def normalize_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(interactions.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = interactions.loc[:, REQUIRED_COLUMNS].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if frame.empty:
        raise ValueError("interaction table is empty after cleaning")
    if not frame["rating"].between(1, 5).all():
        raise ValueError("ratings must be in the 1-5 range")
    frame["timestamp"] = frame["timestamp"].astype(int)
    frame = frame.sort_values(["user_id", "timestamp", "item_id"])
    return frame.drop_duplicates(
        subset=["user_id", "item_id"],
        keep="last",
    ).reset_index(drop=True)


def filter_min_interactions(
    interactions: pd.DataFrame,
    min_interactions: int = 3,
) -> pd.DataFrame:
    if min_interactions < 2:
        raise ValueError("min_interactions must be at least 2")
    frame = normalize_interactions(interactions)
    counts = frame.groupby("user_id")["item_id"].transform("count")
    filtered = frame[counts >= min_interactions].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("no users remain after min_interactions filtering")
    return filtered


def temporal_user_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = filter_min_interactions(interactions, min_interactions=3)
    train_rows = []
    validation_rows = []
    test_rows = []
    for _, group in frame.groupby("user_id", sort=False):
        ordered = group.sort_values("timestamp")
        train_rows.append(ordered.iloc[:-2])
        validation_rows.append(ordered.iloc[-2:-1])
        test_rows.append(ordered.iloc[-1:])
    return (
        pd.concat(train_rows, ignore_index=True),
        pd.concat(validation_rows, ignore_index=True),
        pd.concat(test_rows, ignore_index=True),
    )


def encode_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> RecommendationData:
    combined = pd.concat([train, validation, test], ignore_index=True)
    user_ids = sorted(combined["user_id"].astype(str).unique())
    item_ids = sorted(combined["item_id"].astype(str).unique())
    user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_mapping = {item_id: idx for idx, item_id in enumerate(item_ids)}

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        encoded["user_idx"] = encoded["user_id"].map(user_mapping).astype(int)
        encoded["item_idx"] = encoded["item_id"].map(item_mapping).astype(int)
        encoded["rating"] = encoded["rating"].astype(float)
        return encoded.reset_index(drop=True)

    item_titles = {idx: item_id for item_id, idx in item_mapping.items()}
    return RecommendationData(
        train=encode(train),
        validation=encode(validation),
        test=encode(test),
        n_users=len(user_mapping),
        n_items=len(item_mapping),
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        item_titles=item_titles,
    )


def prepare_recommendation_data(
    dataset: str = "synthetic",
    data_dir: Path = Path("data"),
    users: int = 100,
    items: int = 150,
    density: float = 0.08,
    latent_dim: int = 12,
    seed: int = 42,
) -> RecommendationData:
    if dataset == "synthetic":
        interactions = generate_synthetic_ratings(users, items, density, latent_dim, seed)
    elif dataset == "movielens-100k":
        interactions = load_movielens_100k(data_dir)
    else:
        raise ValueError("dataset must be synthetic or movielens-100k")
    return encode_splits(*temporal_user_split(interactions))


def user_seen_items(interactions: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for user_idx, group in interactions.groupby("user_idx"):
        seen[int(user_idx)] = set(int(value) for value in group["item_idx"])
    return seen


def relevant_items_by_user(
    interactions: pd.DataFrame,
    min_rating: float = 4.0,
) -> dict[int, set[int]]:
    positives = interactions[interactions["rating"] >= min_rating]
    return user_seen_items(positives)
