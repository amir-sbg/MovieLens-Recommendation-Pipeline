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
    user_seen_items,
)
from recsys_lab.metrics import catalog_coverage, personalization, ranking_metrics, rating_metrics
from recsys_lab.models import ItemKNNRecommender, MatrixFactorizationRecommender, PopularityRecommender


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
    knn_neighbors: int = 30
    mf_factors: int = 24
    mf_epochs: int = 15
    mf_learning_rate: float = 0.03
    mf_regularization: float = 0.03

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
    validation_rating: dict[str, float],
    test_rating: dict[str, float],
    ranking: dict[str, float],
    coverage: float,
    diversity: float,
    fit_seconds: float,
) -> dict[str, float | str]:
    return {
        "model": model_name,
        "validation_rmse": validation_rating["rmse"],
        "validation_mae": validation_rating["mae"],
        "test_rmse": test_rating["rmse"],
        "test_mae": test_rating["mae"],
        "recall_at_k": ranking["recall_at_k"],
        "map_at_k": ranking["map_at_k"],
        "ndcg_at_k": ranking["ndcg_at_k"],
        "hit_rate": ranking["hit_rate"],
        "catalog_coverage": coverage,
        "personalization": diversity,
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
        "| model | val RMSE | test RMSE | Recall@K | MAP@K | NDCG@K | coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            "| {model} | {validation_rmse:.4f} | {test_rmse:.4f} | {recall_at_k:.4f} | "
            "{map_at_k:.4f} | {ndcg_at_k:.4f} | {catalog_coverage:.4f} |".format(**row)
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
    heldout = user_seen_items(data.test)
    eval_users = sorted(heldout)

    models = {
        "popularity": PopularityRecommender(),
        "item_knn": ItemKNNRecommender(n_neighbors=config.knn_neighbors),
        "matrix_factorization": MatrixFactorizationRecommender(
            factors=config.mf_factors,
            epochs=config.mf_epochs,
            learning_rate=config.mf_learning_rate,
            regularization=config.mf_regularization,
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
        row = _model_rows(
            model_name=model_name,
            validation_rating=_predict_frame(model, data.validation),
            test_rating=_predict_frame(model, data.test),
            ranking=rank,
            coverage=catalog_coverage(recommendations, data.n_items),
            diversity=personalization(recommendations),
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

    metrics_frame = pd.DataFrame(metric_rows).sort_values("recall_at_k", ascending=False)
    metrics_frame.to_csv(config.report_dir / "model_metrics.csv", index=False)
    pd.DataFrame(mf_model.history).to_csv(config.report_dir / "mf_history.csv", index=False)
    pd.DataFrame(recommendation_rows).to_csv(config.report_dir / "sample_recommendations.csv", index=False)

    summary = {
        "config": config.to_json_dict(),
        "data": {
            "n_users": data.n_users,
            "n_items": data.n_items,
            "train_interactions": len(data.train),
            "validation_interactions": len(data.validation),
            "test_interactions": len(data.test),
        },
        "best_model_by_recall": str(metrics_frame.iloc[0]["model"]),
        "metrics": metrics_frame.to_dict(orient="records"),
        "outputs": {
            "metrics": str(config.report_dir / "model_metrics.csv"),
            "recommendations": str(config.report_dir / "sample_recommendations.csv"),
            "history": str(config.report_dir / "mf_history.csv"),
            "model": str(config.artifact_dir / "matrix_factorization.npz"),
            "markdown_report": str(config.report_dir / "experiment_report.md"),
        },
    }
    with (config.report_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _write_markdown_report(summary, config.report_dir / "experiment_report.md")
    return summary
