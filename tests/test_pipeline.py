from __future__ import annotations

import json

import pandas as pd

from recsys_lab.pipeline import ExperimentConfig, run_experiment


def test_pipeline_writes_reports_and_artifacts(tmp_path) -> None:
    config = ExperimentConfig(
        users=18,
        items=35,
        density=0.25,
        seed=21,
        top_k=5,
        mf_factors=6,
        mf_epochs=3,
        artifact_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
    )

    summary = run_experiment(config)

    metrics_path = tmp_path / "reports" / "model_metrics.csv"
    history_path = tmp_path / "reports" / "mf_history.csv"
    sample_path = tmp_path / "reports" / "sample_recommendations.csv"
    model_path = tmp_path / "artifacts" / "matrix_factorization.npz"
    summary_path = tmp_path / "reports" / "run_summary.json"
    report_path = tmp_path / "reports" / "experiment_report.md"

    assert metrics_path.exists()
    assert history_path.exists()
    assert sample_path.exists()
    assert model_path.exists()
    assert summary_path.exists()
    assert report_path.exists()
    assert summary["data"]["n_users"] == 18
    assert summary["data"]["relevance_threshold"] == 4.0
    assert summary["data"]["ranking_users"] <= summary["data"]["test_interactions"]

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["model"]) == {"popularity", "item_knn", "matrix_factorization"}
    assert {
        "recall_at_k",
        "ranking_users",
        "catalog_coverage",
        "novelty_at_k",
        "long_tail_share_at_k",
        "tail_recall_at_k",
        "head_recall_at_k",
        "recall_at_k_ci_low",
        "recall_at_k_ci_high",
        "exposure_gini",
        "exposure_entropy",
        "fit_seconds",
    }.issubset(metrics.columns)

    with summary_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["best_model_by_recall"] in set(metrics["model"])
    assert "Recommendation Experiment Report" in report_path.read_text(encoding="utf-8")


def test_pipeline_validates_relevance_threshold(tmp_path) -> None:
    config = ExperimentConfig(
        users=8,
        items=20,
        density=0.3,
        relevance_threshold=5.5,
        artifact_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
    )

    try:
        run_experiment(config)
    except ValueError as error:
        assert "relevance_threshold" in str(error)
    else:
        raise AssertionError("invalid relevance threshold was accepted")
