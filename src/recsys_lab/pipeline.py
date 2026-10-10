from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from recsys_lab.data import (
    RecommendationData,
    prepare_recommendation_data,
    relevant_items_by_user,
    user_seen_items,
)
from recsys_lab.metrics import (
    bootstrap_ranking_intervals,
    catalog_coverage,
    exposure_gini,
    long_tail_share_at_k,
    novelty_at_k,
    normalized_exposure_entropy,
    personalization,
    ranking_metrics,
    recall_by_popularity_segment,
    rating_metrics,
)
from recsys_lab.models import (
    BPRMatrixFactorizationRecommender,
    ItemKNNRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender,
)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "synthetic"
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")
    users: int = 100
    items: int = 150
    density: float = 0.08
    latent_dim: int = 12
    seed: int = 42
    top_k: int = 10
    relevance_threshold: float = 4.0
    knn_neighbors: int = 30
    mf_factors: int = 24
    mf_epochs: int = 15
    mf_learning_rate: float = 0.03
    mf_regularization: float = 0.03
    mf_patience: int | None = 5
    bpr_factors: int = 24
    bpr_epochs: int = 15
    bpr_learning_rate: float = 0.03
    bpr_regularization: float = 0.01
    bootstrap_resamples: int = 300

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ExperimentConfig":
        path_fields = {"data_dir", "artifact_dir", "report_dir"}
        cleaned = {
            key: Path(value) if key in path_fields else value
            for key, value in values.items()
        }
        return cls(**cleaned)

    def to_json_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("data_dir", "artifact_dir", "report_dir"):
            values[key] = str(values[key])
        return values


def _predict_frame(model: object, frame: pd.DataFrame) -> dict[str, float]:
    predictions = model.predict_pairs(
        frame["user_idx"].to_numpy(dtype=int),
        frame["item_idx"].to_numpy(dtype=int),
    )
    return rating_metrics(frame["rating"].to_numpy(dtype=float), predictions)


def _recommend_for_users(
    model: object,
    users: list[int],
    seen: dict[int, set[int]],
    top_k: int,
) -> dict[int, list[int]]:
    return {
        user: model.recommend(user, seen_items=seen.get(user, set()), k=top_k)
        for user in users
    }


def _model_rows(
    model_name: str,
    validation_rating: dict[str, float] | None,
    test_rating: dict[str, float] | None,
    ranking: dict[str, float],
    ranking_intervals: dict[str, float],
    segment_recall: dict[str, float],
    coverage: float,
    diversity: float,
    exposure_gini_value: float,
    exposure_entropy: float,
    novelty: float,
    long_tail_share: float,
    fit_seconds: float,
) -> dict[str, float | str | None]:
    return {
        "model": model_name,
        "validation_rmse": validation_rating["rmse"] if validation_rating else None,
        "validation_mae": validation_rating["mae"] if validation_rating else None,
        "test_rmse": test_rating["rmse"] if test_rating else None,
        "test_mae": test_rating["mae"] if test_rating else None,
        "ranking_users": ranking["users"],
        "recall_at_k": ranking["recall_at_k"],
        "map_at_k": ranking["map_at_k"],
        "mrr_at_k": ranking["mrr_at_k"],
        "ndcg_at_k": ranking["ndcg_at_k"],
        "hit_rate": ranking["hit_rate"],
        **ranking_intervals,
        **segment_recall,
        "catalog_coverage": coverage,
        "personalization": diversity,
        "exposure_gini": exposure_gini_value,
        "exposure_entropy": exposure_entropy,
        "novelty_at_k": novelty,
        "long_tail_share_at_k": long_tail_share,
        "fit_seconds": fit_seconds,
    }


def _item_names(data: RecommendationData, items: list[int]) -> str:
    return " | ".join(data.item_titles.get(item, str(item)) for item in items)


def _write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Recommendation Experiment Report",
        "",
        f"Best model by Recall@K: `{summary['best_model_by_recall']}`",
        "",
        "## Data",
        "",
        f"- Users: {summary['data']['n_users']}",
        f"- Items: {summary['data']['n_items']}",
        f"- Train interactions: {summary['data']['train_interactions']}",
        f"- Validation interactions: {summary['data']['validation_interactions']}",
        f"- Test interactions: {summary['data']['test_interactions']}",
        "",
        "## Model comparison",
        "",
        "| model | test RMSE | Recall@K (95% CI) | NDCG@K | tail recall | exposure Gini |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        test_rmse = "n/a" if row["test_rmse"] is None else f"{row['test_rmse']:.4f}"
        lines.append(
            f"| {row['model']} | {test_rmse} | {row['recall_at_k']:.4f} "
            f"[{row['recall_at_k_ci_low']:.4f}, {row['recall_at_k_ci_high']:.4f}] | "
            f"{row['ndcg_at_k']:.4f} | {row['tail_recall_at_k']:.4f} | "
            f"{row['exposure_gini']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The ranking metrics use each user's final held-out interaction, while validation RMSE tracks",
            "explicit-rating fit during model selection.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    if not 1.0 <= config.relevance_threshold <= 5.0:
        raise ValueError("relevance_threshold must be in the 1-5 rating range")
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_recommendation_data(
        dataset=config.dataset,
        data_dir=config.data_dir,
        users=config.users,
        items=config.items,
        density=config.density,
        latent_dim=config.latent_dim,
        seed=config.seed,
    )
    seen = user_seen_items(data.train)
    heldout = relevant_items_by_user(data.test, min_rating=config.relevance_threshold)
    eval_users = sorted(heldout)
    item_popularity = {
        int(item): int(count)
        for item, count in data.train.groupby("item_idx")["rating"].count().items()
    }
    total_interactions = int(len(data.train))

    models = {
        "popularity": PopularityRecommender(),
        "item_knn": ItemKNNRecommender(n_neighbors=config.knn_neighbors),
        "matrix_factorization": MatrixFactorizationRecommender(
            factors=config.mf_factors,
            epochs=config.mf_epochs,
            learning_rate=config.mf_learning_rate,
            regularization=config.mf_regularization,
            patience=config.mf_patience,
            seed=config.seed,
        ),
        "bpr_matrix_factorization": BPRMatrixFactorizationRecommender(
            factors=config.bpr_factors,
            epochs=config.bpr_epochs,
            learning_rate=config.bpr_learning_rate,
            regularization=config.bpr_regularization,
            positive_threshold=config.relevance_threshold,
            seed=config.seed,
        ),
    }

    metric_rows = []
    recommendation_rows = []
    fitted_models: dict[str, object] = {}
    for model_name, model in models.items():
        started = time.perf_counter()
        if model_name == "matrix_factorization":
            model.fit(data.train, validation=data.validation, n_users=data.n_users, n_items=data.n_items)
        else:
            model.fit(data.train, n_users=data.n_users, n_items=data.n_items)
        fit_seconds = time.perf_counter() - started
        fitted_models[model_name] = model

        recommendations = _recommend_for_users(model, eval_users, seen, config.top_k)
        rank = ranking_metrics(recommendations, heldout, k=config.top_k)
        intervals = bootstrap_ranking_intervals(
            recommendations,
            heldout,
            k=config.top_k,
            n_resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        segment_recall = recall_by_popularity_segment(
            recommendations,
            heldout,
            item_popularity,
            k=config.top_k,
        )
        row = _model_rows(
            model_name=model_name,
            validation_rating=None if model_name == "bpr_matrix_factorization" else _predict_frame(model, data.validation),
            test_rating=None if model_name == "bpr_matrix_factorization" else _predict_frame(model, data.test),
            ranking=rank,
            ranking_intervals=intervals,
            segment_recall=segment_recall,
            coverage=catalog_coverage(recommendations, data.n_items),
            diversity=personalization(recommendations),
            exposure_gini_value=exposure_gini(recommendations, data.n_items, config.top_k),
            exposure_entropy=normalized_exposure_entropy(
                recommendations, data.n_items, config.top_k
            ),
            novelty=novelty_at_k(
                recommendations,
                item_popularity,
                total_interactions,
                config.top_k,
            ),
            long_tail_share=long_tail_share_at_k(
                recommendations,
                item_popularity,
                k=config.top_k,
            ),
            fit_seconds=fit_seconds,
        )
        metric_rows.append(row)

        for user in eval_users[:8]:
            recommendation_rows.append(
                {
                    "model": model_name,
                    "user_idx": user,
                    "heldout_item": _item_names(data, sorted(heldout[user])),
                    "recommendations": _item_names(data, recommendations[user]),
                }
            )

    mf_model = fitted_models["matrix_factorization"]
    mf_model.save_npz(config.artifact_dir / "matrix_factorization.npz")
    bpr_model = fitted_models["bpr_matrix_factorization"]
    bpr_model.save_npz(config.artifact_dir / "bpr_matrix_factorization.npz")

    metrics_frame = pd.DataFrame(metric_rows).sort_values("recall_at_k", ascending=False)
    metrics_frame.to_csv(config.report_dir / "model_metrics.csv", index=False)
    pd.DataFrame(mf_model.history).to_csv(config.report_dir / "mf_history.csv", index=False)
    pd.DataFrame(bpr_model.history).to_csv(config.report_dir / "bpr_history.csv", index=False)
    pd.DataFrame(recommendation_rows).to_csv(config.report_dir / "sample_recommendations.csv", index=False)

    summary = {
        "config": config.to_json_dict(),
        "data": {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "train_interactions": len(data.train),
            "validation_interactions": len(data.validation),
            "test_interactions": len(data.test),
            "ranking_users": len(eval_users),
            "relevance_threshold": config.relevance_threshold,
        },
        "best_model_by_recall": str(metrics_frame.iloc[0]["model"]),
        "metrics": metrics_frame.astype(object).where(pd.notna(metrics_frame), None).to_dict(orient="records"),
        "outputs": {
            "metrics": str(config.report_dir / "model_metrics.csv"),
            "recommendations": str(config.report_dir / "sample_recommendations.csv"),
            "history": str(config.report_dir / "mf_history.csv"),
            "bpr_history": str(config.report_dir / "bpr_history.csv"),
            "model": str(config.artifact_dir / "matrix_factorization.npz"),
            "bpr_model": str(config.artifact_dir / "bpr_matrix_factorization.npz"),
            "markdown_report": str(config.report_dir / "experiment_report.md"),
        },
    }
    with (config.report_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_markdown_report(summary, config.report_dir / "experiment_report.md")
    return summary
