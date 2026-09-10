from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys_lab.pipeline import ExperimentConfig, run_experiment


def _run_from_args(args: argparse.Namespace) -> dict[str, object]:
    config = ExperimentConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        users=args.users,
        items=args.items,
        density=args.density,
        latent_dim=args.latent_dim,
        seed=args.seed,
        top_k=args.top_k,
        knn_neighbors=args.knn_neighbors,
        mf_factors=args.mf_factors,
        mf_epochs=args.mf_epochs,
        mf_learning_rate=args.mf_learning_rate,
        mf_regularization=args.mf_regularization,
        mf_patience=args.mf_patience,
    )
    return run_experiment(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run recommendation-system experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from command-line options.")
    run_parser.add_argument("--dataset", choices=["synthetic", "movielens-100k"], default="synthetic")
    run_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    run_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    run_parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    run_parser.add_argument("--users", type=int, default=100)
    run_parser.add_argument("--items", type=int, default=150)
    run_parser.add_argument("--density", type=float, default=0.08)
    run_parser.add_argument("--latent-dim", type=int, default=12)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--top-k", type=int, default=10)
    run_parser.add_argument("--knn-neighbors", type=int, default=30)
    run_parser.add_argument("--mf-factors", type=int, default=24)
    run_parser.add_argument("--mf-epochs", type=int, default=15)
    run_parser.add_argument("--mf-learning-rate", type=float, default=0.03)
    run_parser.add_argument("--mf-regularization", type=float, default=0.03)
    run_parser.add_argument("--mf-patience", type=int, default=5)
    run_parser.set_defaults(func=_run_from_args)

    config_parser = subparsers.add_parser("run-config", help="Run an experiment from a JSON config.")
    config_parser.add_argument("path", type=Path)
    config_parser.set_defaults(func=_run_from_config)
    return parser


def _run_from_config(args: argparse.Namespace) -> dict[str, object]:
    with args.path.open("r", encoding="utf-8") as handle:
        config = ExperimentConfig.from_mapping(json.load(handle))
    return run_experiment(config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = args.func(args)
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
