from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_lab.metrics import rmse


def _infer_shape(interactions: pd.DataFrame, n_users: int | None, n_items: int | None) -> tuple[int, int]:
    if interactions.empty:
        raise ValueError("training interactions cannot be empty")
    users = n_users if n_users is not None else int(interactions["user_idx"].max()) + 1
    items = n_items if n_items is not None else int(interactions["item_idx"].max()) + 1
    return users, items


def _best_items_from_scores(scores: np.ndarray, seen_items: set[int], k: int) -> list[int]:
    if k < 1:
        raise ValueError("k must be positive")
    masked = scores.astype(float, copy=True)
    if seen_items:
        blocked = [item for item in seen_items if 0 <= item < len(masked)]
        masked[blocked] = -np.inf
    finite = np.flatnonzero(np.isfinite(masked))
    if len(finite) == 0:
        return []
    ranked = finite[np.argsort(masked[finite])[::-1]]
    return [int(item) for item in ranked[:k]]


@dataclass
class PopularityRecommender:
    prior_weight: float = 12.0
    item_scores: np.ndarray | None = None
    item_means: np.ndarray | None = None
    item_counts: np.ndarray | None = None
    global_mean: float = 3.0

    def fit(
        self,
        interactions: pd.DataFrame,
        n_users: int | None = None,
        n_items: int | None = None,
    ) -> "PopularityRecommender":
        if self.prior_weight < 0:
            raise ValueError("prior_weight must not be negative")
        _, items = _infer_shape(interactions, n_users, n_items)
        self.global_mean = float(interactions["rating"].mean())
        self.item_means = np.full(items, self.global_mean, dtype=float)
        self.item_counts = np.zeros(items, dtype=float)

        grouped = interactions.groupby("item_idx")["rating"].agg(["sum", "count"])
        for item_idx, row in grouped.iterrows():
            idx = int(item_idx)
            count = float(row["count"])
            self.item_means[idx] = float(
                (row["sum"] + self.prior_weight * self.global_mean)
                / (count + self.prior_weight)
            )
            self.item_counts[idx] = float(row["count"])

        confidence = np.log1p(self.item_counts)
        self.item_scores = self.item_means + 0.06 * confidence
        return self

    def predict_pairs(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        if self.item_means is None:
            raise RuntimeError("fit the popularity recommender before prediction")
        item_indices = np.asarray(item_indices, dtype=int)
        predictions = np.full(len(item_indices), self.global_mean, dtype=float)
        known = (item_indices >= 0) & (item_indices < len(self.item_means))
        predictions[known] = self.item_means[item_indices[known]]
        return np.clip(predictions, 1.0, 5.0)

    def recommend(self, user_idx: int, seen_items: set[int] | None = None, k: int = 10) -> list[int]:
        if self.item_scores is None:
            raise RuntimeError("fit the popularity recommender before recommendation")
        return _best_items_from_scores(self.item_scores, seen_items or set(), k)


@dataclass
class ItemKNNRecommender:
    n_neighbors: int = 40
    shrinkage: float = 20.0
    similarity: np.ndarray | None = None
    centered: np.ndarray | None = None
    observed: np.ndarray | None = None
    item_means: np.ndarray | None = None
    global_mean: float = 3.0
    fallback: PopularityRecommender = field(default_factory=PopularityRecommender)

    def fit(
        self,
        interactions: pd.DataFrame,
        n_users: int | None = None,
        n_items: int | None = None,
    ) -> "ItemKNNRecommender":
        users, items = _infer_shape(interactions, n_users, n_items)
        self.global_mean = float(interactions["rating"].mean())
        self.fallback.fit(interactions, n_users=users, n_items=items)
        self.item_means = self.fallback.item_means.copy()

        ratings = np.zeros((users, items), dtype=float)
        self.observed = np.zeros((users, items), dtype=bool)
        for row in interactions.itertuples(index=False):
            ratings[int(row.user_idx), int(row.item_idx)] = float(row.rating)
            self.observed[int(row.user_idx), int(row.item_idx)] = True

        self.centered = np.where(self.observed, ratings - self.global_mean, 0.0)
        norms = np.linalg.norm(self.centered, axis=0)
        safe = np.where(norms > 0, norms, 1.0)
        normalized = self.centered / safe
        similarity = normalized.T @ normalized

        co_counts = self.observed.astype(float).T @ self.observed.astype(float)
        similarity *= co_counts / (co_counts + self.shrinkage)
        np.fill_diagonal(similarity, 0.0)

        if self.n_neighbors < items:
            pruned = np.zeros_like(similarity)
            for item_idx in range(items):
                row = similarity[item_idx]
                keep = np.argpartition(np.abs(row), -self.n_neighbors)[-self.n_neighbors :]
                pruned[item_idx, keep] = row[keep]
            similarity = pruned

        self.similarity = similarity
        return self

    def scores_for_user(self, user_idx: int) -> np.ndarray:
        if self.similarity is None or self.centered is None or self.observed is None or self.item_means is None:
            raise RuntimeError("fit the item KNN recommender before scoring")
        if user_idx < 0 or user_idx >= self.centered.shape[0]:
            return self.fallback.item_scores.copy()

        rated = self.observed[user_idx]
        if not rated.any():
            return self.fallback.item_scores.copy()

        numerator = self.centered[user_idx] @ self.similarity
        denominator = np.abs(self.similarity[rated]).sum(axis=0)
        correction = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)
        scores = self.item_means + correction
        return np.clip(scores, 1.0, 5.0)

    def predict_pairs(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        user_indices = np.asarray(user_indices, dtype=int)
        item_indices = np.asarray(item_indices, dtype=int)
        predictions = []
        for user_idx, item_idx in zip(user_indices, item_indices):
            scores = self.scores_for_user(int(user_idx))
            if 0 <= item_idx < len(scores):
                predictions.append(float(scores[item_idx]))
            else:
                predictions.append(self.global_mean)
        return np.asarray(predictions, dtype=float)

    def recommend(self, user_idx: int, seen_items: set[int] | None = None, k: int = 10) -> list[int]:
        return _best_items_from_scores(self.scores_for_user(user_idx), seen_items or set(), k)


@dataclass
class MatrixFactorizationRecommender:
    factors: int = 24
    epochs: int = 20
    learning_rate: float = 0.03
    regularization: float = 0.03
    patience: int | None = 5
    min_delta: float = 1e-4
    seed: int = 42
    global_mean: float = 3.0
    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    user_bias: np.ndarray | None = None
    item_bias: np.ndarray | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def fit(
        self,
        interactions: pd.DataFrame,
        validation: pd.DataFrame | None = None,
        n_users: int | None = None,
        n_items: int | None = None,
    ) -> "MatrixFactorizationRecommender":
        self._validate_training_settings()
        users, items = _infer_shape(interactions, n_users, n_items)
        rng = np.random.default_rng(self.seed)
        self.global_mean = float(interactions["rating"].mean())
        self.user_factors = rng.normal(0, 0.05, size=(users, self.factors))
        self.item_factors = rng.normal(0, 0.05, size=(items, self.factors))
        self.user_bias = np.zeros(users, dtype=float)
        self.item_bias = np.zeros(items, dtype=float)
        self.history = []

        train_users = interactions["user_idx"].to_numpy(dtype=int)
        train_items = interactions["item_idx"].to_numpy(dtype=int)
        train_ratings = interactions["rating"].to_numpy(dtype=float)
        best_validation_rmse = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(interactions))
            for idx in order:
                user = train_users[idx]
                item = train_items[idx]
                rating = train_ratings[idx]
                prediction = self._raw_predict(user, item)
                error = rating - prediction

                old_user = self.user_factors[user].copy()
                old_item = self.item_factors[item].copy()
                self.user_bias[user] += self.learning_rate * (
                    error - self.regularization * self.user_bias[user]
                )
                self.item_bias[item] += self.learning_rate * (
                    error - self.regularization * self.item_bias[item]
                )
                self.user_factors[user] += self.learning_rate * (
                    error * old_item - self.regularization * old_user
                )
                self.item_factors[item] += self.learning_rate * (
                    error * old_user - self.regularization * old_item
                )

            row = {
                "epoch": float(epoch),
                "train_rmse": rmse(train_ratings, self.predict_pairs(train_users, train_items)),
            }
            if validation is not None and not validation.empty:
                row["validation_rmse"] = rmse(
                    validation["rating"].to_numpy(dtype=float),
                    self.predict_pairs(
                        validation["user_idx"].to_numpy(dtype=int),
                        validation["item_idx"].to_numpy(dtype=int),
                    ),
                )
                if row["validation_rmse"] < best_validation_rmse - self.min_delta:
                    best_validation_rmse = row["validation_rmse"]
                    best_state = self._state_dict()
                    wait = 0
                else:
                    wait += 1
            self.history.append(row)
            if self.patience is not None and validation is not None and wait >= self.patience:
                break
        if best_state is not None:
            self._load_state_dict(best_state)
        return self

    def _raw_predict(self, user_idx: int, item_idx: int) -> float:
        if (
            self.user_factors is None
            or self.item_factors is None
            or self.user_bias is None
            or self.item_bias is None
        ):
            raise RuntimeError("fit the matrix factorization model before prediction")
        return float(
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + self.user_factors[user_idx] @ self.item_factors[item_idx]
        )

    def predict_pairs(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        user_indices = np.asarray(user_indices, dtype=int)
        item_indices = np.asarray(item_indices, dtype=int)
        if user_indices.shape != item_indices.shape:
            raise ValueError("user_indices and item_indices must have the same shape")
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("fit the matrix factorization model before prediction")
        predictions = []
        for user, item in zip(user_indices, item_indices):
            if 0 <= user < len(self.user_factors) and 0 <= item < len(self.item_factors):
                predictions.append(self._raw_predict(int(user), int(item)))
            elif 0 <= item < len(self.item_factors):
                predictions.append(self.global_mean + self.item_bias[item])
            else:
                predictions.append(self.global_mean)
        return np.clip(np.asarray(predictions, dtype=float), 1.0, 5.0)

    def scores_for_user(self, user_idx: int) -> np.ndarray:
        if (
            self.user_factors is None
            or self.item_factors is None
            or self.user_bias is None
            or self.item_bias is None
        ):
            raise RuntimeError("fit the matrix factorization model before scoring")
        if not 0 <= user_idx < len(self.user_factors):
            return np.clip(self.global_mean + self.item_bias, 1.0, 5.0)
        scores = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.item_factors @ self.user_factors[user_idx]
        )
        return np.clip(scores, 1.0, 5.0)

    def recommend(self, user_idx: int, seen_items: set[int] | None = None, k: int = 10) -> list[int]:
        return _best_items_from_scores(self.scores_for_user(user_idx), seen_items or set(), k)

    def save_npz(self, path: Path) -> None:
        if (
            self.user_factors is None
            or self.item_factors is None
            or self.user_bias is None
            or self.item_bias is None
        ):
            raise RuntimeError("fit the matrix factorization model before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            global_mean=np.array([self.global_mean]),
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            user_bias=self.user_bias,
            item_bias=self.item_bias,
        )

    @classmethod
    def load_npz(cls, path: Path) -> "MatrixFactorizationRecommender":
        with np.load(path) as payload:
            required = {"global_mean", "user_factors", "item_factors", "user_bias", "item_bias"}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"matrix factorization artifact is missing: {sorted(missing)}")
            user_factors = np.asarray(payload["user_factors"], dtype=float)
            item_factors = np.asarray(payload["item_factors"], dtype=float)
            user_bias = np.asarray(payload["user_bias"], dtype=float)
            item_bias = np.asarray(payload["item_bias"], dtype=float)
            global_mean = float(np.asarray(payload["global_mean"]).reshape(-1)[0])

        if user_factors.ndim != 2 or item_factors.ndim != 2:
            raise ValueError("factor arrays must be 2-D")
        if user_factors.shape[1] != item_factors.shape[1]:
            raise ValueError("user and item factors must use the same number of factors")
        if user_bias.shape != (user_factors.shape[0],):
            raise ValueError("user bias shape does not match user factors")
        if item_bias.shape != (item_factors.shape[0],):
            raise ValueError("item bias shape does not match item factors")

        model = cls(factors=user_factors.shape[1], epochs=1)
        model.global_mean = global_mean
        model.user_factors = user_factors
        model.item_factors = item_factors
        model.user_bias = user_bias
        model.item_bias = item_bias
        return model

    def _validate_training_settings(self) -> None:
        if self.factors < 1 or self.epochs < 1:
            raise ValueError("factors and epochs must be positive")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must not be negative")
        if self.regularization < 0:
            raise ValueError("regularization must not be negative")
        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be positive")
        if self.min_delta < 0:
            raise ValueError("min_delta must not be negative")

    def _state_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "global_mean": float(self.global_mean),
            "user_factors": self.user_factors.copy(),
            "item_factors": self.item_factors.copy(),
            "user_bias": self.user_bias.copy(),
            "item_bias": self.item_bias.copy(),
        }

    def _load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        self.global_mean = float(state["global_mean"])
        self.user_factors = np.asarray(state["user_factors"], dtype=float).copy()
        self.item_factors = np.asarray(state["item_factors"], dtype=float).copy()
        self.user_bias = np.asarray(state["user_bias"], dtype=float).copy()
        self.item_bias = np.asarray(state["item_bias"], dtype=float).copy()


@dataclass
class BPRMatrixFactorizationRecommender:
    factors: int = 24
    epochs: int = 15
    learning_rate: float = 0.03
    regularization: float = 0.01
    positive_threshold: float = 4.0
    samples_per_epoch: int | None = None
    seed: int = 42
    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    item_bias: np.ndarray | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def fit(
        self,
        interactions: pd.DataFrame,
        n_users: int | None = None,
        n_items: int | None = None,
    ) -> "BPRMatrixFactorizationRecommender":
        self._validate_training_settings()
        users, items = _infer_shape(interactions, n_users, n_items)
        positives = interactions[interactions["rating"] >= self.positive_threshold]
        positive_by_user = {
            int(user): np.asarray(group["item_idx"].unique(), dtype=int)
            for user, group in positives.groupby("user_idx")
        }
        observed_by_user = {
            int(user): set(int(item) for item in group["item_idx"])
            for user, group in interactions.groupby("user_idx")
        }
        eligible_users = np.asarray(
            [
                user
                for user, user_positives in positive_by_user.items()
                if len(user_positives) > 0 and len(observed_by_user[user]) < items
            ],
            dtype=int,
        )
        if eligible_users.size == 0:
            raise ValueError("BPR needs at least one positive interaction and one unobserved item")

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0.0, 0.05, size=(users, self.factors))
        self.item_factors = rng.normal(0.0, 0.05, size=(items, self.factors))
        self.item_bias = np.zeros(items, dtype=float)
        self.history = []
        draws = self.samples_per_epoch or int(sum(len(values) for values in positive_by_user.values()))

        for epoch in range(1, self.epochs + 1):
            losses = []
            for _ in range(draws):
                user = int(rng.choice(eligible_users))
                positive = int(rng.choice(positive_by_user[user]))
                negative = int(rng.integers(items))
                while negative in observed_by_user[user]:
                    negative = int(rng.integers(items))

                user_vector = self.user_factors[user].copy()
                positive_vector = self.item_factors[positive].copy()
                negative_vector = self.item_factors[negative].copy()
                score_difference = float(
                    self.item_bias[positive]
                    - self.item_bias[negative]
                    + user_vector @ (positive_vector - negative_vector)
                )
                gradient = 1.0 / (1.0 + np.exp(np.clip(score_difference, -35.0, 35.0)))

                self.user_factors[user] += self.learning_rate * (
                    gradient * (positive_vector - negative_vector)
                    - self.regularization * user_vector
                )
                self.item_factors[positive] += self.learning_rate * (
                    gradient * user_vector - self.regularization * positive_vector
                )
                self.item_factors[negative] += self.learning_rate * (
                    -gradient * user_vector - self.regularization * negative_vector
                )
                self.item_bias[positive] += self.learning_rate * (
                    gradient - self.regularization * self.item_bias[positive]
                )
                self.item_bias[negative] += self.learning_rate * (
                    -gradient - self.regularization * self.item_bias[negative]
                )
                losses.append(float(np.logaddexp(0.0, -score_difference)))

            self.history.append(
                {
                    "epoch": float(epoch),
                    "pairwise_loss": float(np.mean(losses)),
                }
            )
        return self

    def scores_for_user(self, user_idx: int) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None or self.item_bias is None:
            raise RuntimeError("fit the BPR model before scoring")
        if not 0 <= user_idx < len(self.user_factors):
            return self.item_bias.copy()
        return self.item_bias + self.item_factors @ self.user_factors[user_idx]

    def recommend(self, user_idx: int, seen_items: set[int] | None = None, k: int = 10) -> list[int]:
        return _best_items_from_scores(self.scores_for_user(user_idx), seen_items or set(), k)

    def save_npz(self, path: Path) -> None:
        if self.user_factors is None or self.item_factors is None or self.item_bias is None:
            raise RuntimeError("fit the BPR model before saving")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            item_bias=self.item_bias,
            positive_threshold=np.array([self.positive_threshold]),
        )

    def _validate_training_settings(self) -> None:
        if self.factors < 1 or self.epochs < 1:
            raise ValueError("factors and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.regularization < 0:
            raise ValueError("regularization must not be negative")
        if not 1.0 <= self.positive_threshold <= 5.0:
            raise ValueError("positive_threshold must be in the 1-5 rating range")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
