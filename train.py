"""Train a fraud model on the frozen split and save it as an artifact.

Usage:
    python train.py <train_csv> <out_dir> [--class-weight balanced]

Fits a sklearn Pipeline (scaler + model fused) and writes:
    model.joblib     - the trained pipeline: raw features in, probability out
    train_metadata.json  - config + provenance (git commit, timing, data counts)
Metrics are evaluate.py's job.
"""
import json
import subprocess
import time
from functools import partial
from pathlib import Path

import click
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import features


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


@click.command(help=__doc__)
@click.argument("train_csv", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(path_type=Path))
@click.option("--class-weight", type=click.Choice(["none", "balanced"]),
              default="none", show_default=True,
              help="'balanced' weights each class inversely to its frequency.")
@click.option("--features", "feature_families", multiple=True,
              type=click.Choice(features.FAMILIES),
              help="Feature families to add (repeatable): --features time --features interactions")
@click.option("--penalty", type=click.Choice(["l2", "l1"]), default="l2",
              show_default=True,
              help="l1 drives useless weights to exactly zero (feature selection).")
@click.option("--model", "model_name", type=click.Choice(["logreg", "xgboost"]),
              default="logreg", show_default=True,
              help="xgboost uses library defaults (100 trees, depth 6) + fixed seed.")
def main(train_csv, out_dir, class_weight, feature_families, penalty, model_name):
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    X, y = train.drop(columns=["Class"]), train["Class"]

    if model_name == "xgboost":
        from xgboost import XGBClassifier
        # Library defaults (Shawn's call): 100 trees, depth 6, lr 0.3.
        # Fixed seed for reproducibility; aucpr matches our headline metric.
        clf = XGBClassifier(n_jobs=-1, random_state=42, eval_metric="aucpr")
        model_config = {"params": "xgboost defaults (100 trees, depth 6, lr 0.3)",
                        "random_state": 42}
    else:
        clf = LogisticRegression(
            max_iter=1000,
            class_weight=None if class_weight == "none" else class_weight,
            penalty=penalty,
            # liblinear for BOTH penalties (Shawn's call 2026-08-03): one
            # optimizer across all runs makes configs comparable; costs speed
            # vs lbfgs on l2.
            solver="liblinear",
        )
        model_config = {"class_weight": class_weight, "penalty": penalty,
                        "max_iter": 1000}

    # One artifact = feature engineering + preprocessing + model. The feature
    # step lives INSIDE the pipeline so the saved model takes raw columns —
    # evaluate.py and any future serving code need zero knowledge of features.
    # (Scaler is a no-op for trees but harmless; kept for uniformity.)
    pipeline = Pipeline([
        ("features", FunctionTransformer(
            partial(features.add, families=list(feature_families)))),
        ("scaler", StandardScaler()),
        ("model", clf),
    ])
    t0 = time.perf_counter()
    pipeline.fit(X, y)
    train_seconds = time.perf_counter() - t0

    joblib.dump(pipeline, out_dir / "model.joblib")

    # Which columns did the model actually use? (L1 zeroes useless weights;
    # tree models have no coefficients — skip.)
    col_names = list(features.add(X.head(1), feature_families).columns)
    fitted = pipeline.named_steps["model"]
    zeroed = ([n for n, c in zip(col_names, fitted.coef_[0]) if c == 0.0]
              if hasattr(fitted, "coef_") else [])

    meta = {
        "model": model_name,
        "config": {
            **model_config,
            "features": sorted(feature_families),
            "n_input_columns": len(col_names),
        },
        "n_zeroed_coefficients": len(zeroed),
        "zeroed_features": zeroed,
        "train_csv": str(train_csv),
        "train_rows": len(train),
        "train_frauds": int(y.sum()),
        "train_seconds": round(train_seconds, 2),
        "git_commit": git_commit(),
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2))
    click.echo(json.dumps(meta, indent=2))
    click.echo(f"saved {out_dir}/model.joblib")


if __name__ == "__main__":
    main()
