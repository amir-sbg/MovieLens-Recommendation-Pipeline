# MovieLens Recommendation Pipeline

A compact recommendation-system project for explicit-feedback movie ratings. The repo walks through the core ML workflow for recommenders: interaction cleaning, temporal train/validation/test splitting, baseline ranking models, matrix factorization, and top-k evaluation.

The goal is not to hide the logic behind a framework. Most of the recommender pieces are implemented directly with NumPy and pandas so the modeling assumptions, ranking behavior, and evaluation tradeoffs stay easy to inspect.

## Pipeline

```text
ratings
  -> cleaning and ID encoding
  -> per-user temporal holdout
  -> popularity baseline
  -> item-item collaborative filtering
  -> matrix factorization with SGD
  -> rating + ranking evaluation
  -> CSV, JSON, markdown, and model artifacts
```

## What is implemented

- Synthetic rating generator for fast local experiments
- Optional MovieLens 100K loader
- Per-user temporal split so validation/test simulate future recommendations
- Popularity recommender with Bayesian count-aware smoothing
- Item-item collaborative filtering with cosine similarity and shrinkage
- Matrix factorization trained with explicit-feedback SGD and validation early stopping
- Ranking metrics: Recall@K, MAP@K, NDCG@K, hit rate, catalog coverage, personalization, novelty, long-tail share
- Rating metrics: RMSE and MAE
- CLI runner, reproducible config, saved reports, and unit tests

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

make smoke
```

Run a slightly larger synthetic experiment:

```bash
python -m recsys_lab.cli run \
  --users 120 \
  --items 180 \
  --density 0.10 \
  --mf-factors 24 \
  --mf-epochs 15 \
  --mf-patience 5 \
  --top-k 10
```

Run from the example config:

```bash
python -m recsys_lab.cli run-config examples/small_run.json
```

Use MovieLens 100K:

```bash
python -m recsys_lab.cli run --dataset movielens-100k --mf-epochs 20 --top-k 10
```

The MovieLens command downloads the public dataset into `data/` on first run.

## Outputs

Each run writes:

- `reports/model_metrics.csv` — model comparison table with rating, ranking, novelty, and long-tail metrics
- `reports/sample_recommendations.csv` — example held-out items and recommended lists
- `reports/mf_history.csv` — matrix-factorization training curve
- `reports/run_summary.json` — config, data shape, metrics, and artifact paths
- `reports/experiment_report.md` — short readable experiment summary
- `artifacts/matrix_factorization.npz` — learned matrix-factorization parameters

`data/`, `reports/`, and `artifacts/` are ignored by git because they are generated locally.

## Tests

```bash
make test
```

The tests cover data splitting, encoding, ranking metrics, baseline recommenders, matrix factorization, and the full experiment runner.
