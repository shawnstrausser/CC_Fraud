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
def main(train_csv, out_dir, class_weight, feature_families):
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    X, y = train.drop(columns=["Class"]), train["Class"]

    # One artifact = feature engineering + preprocessing + model. The feature
    # step lives INSIDE the pipeline so the saved model takes raw columns —
    # evaluate.py and any future serving code need zero knowledge of features.
    pipeline = Pipeline([
        ("features", FunctionTransformer(
            partial(features.add, families=list(feature_families)))),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight=None if class_weight == "none" else class_weight,
        )),
    ])
    t0 = time.perf_counter()
    pipeline.fit(X, y)
    train_seconds = time.perf_counter() - t0

    joblib.dump(pipeline, out_dir / "model.joblib")
    meta = {
        "model": "logistic_regression",
        "config": {
            "class_weight": class_weight,
            "features": sorted(feature_families),
            "n_input_columns": features.add(X.head(1), feature_families).shape[1],
            "max_iter": 1000,
        },
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
