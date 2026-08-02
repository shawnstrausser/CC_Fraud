"""Train a fraud model on the frozen split.

Usage:
    python train.py <train_csv> <output_dir> [--class-weight balanced]

One run = one configuration = one metrics.json (which records the config).
Evaluates on TRAINING data — fit quality, not generalization; the frozen
test.csv stays untouched until final model comparison.
"""
import json
import time
from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

THRESHOLD = 0.5


@click.command(help=__doc__)
@click.argument("train_csv", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(path_type=Path))
@click.option("--class-weight", type=click.Choice(["none", "balanced"]),
              default="none", show_default=True,
              help="'balanced' weights each class inversely to its frequency.")
def main(train_csv, out_dir, class_weight):
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    X_train, y_train = train.drop(columns=["Class"]), train["Class"]

    # Logistic regression needs features on comparable scales to converge.
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        max_iter=1000,
        class_weight=None if class_weight == "none" else class_weight,
    )
    t0 = time.perf_counter()
    model.fit(scaler.transform(X_train), y_train)
    train_seconds = time.perf_counter() - t0

    proba = model.predict_proba(scaler.transform(X_train))[:, 1]
    pred = proba >= THRESHOLD

    metrics = {
        "model": "logistic_regression",
        "config": {
            "class_weight": class_weight,
            "threshold": THRESHOLD,
            "max_iter": 1000,
        },
        "split": "time_based_80_20_frozen",
        "evaluated_on": "train",
        "eval_frauds": int(y_train.sum()),
        "train_seconds": round(train_seconds, 2),
        "roc_auc": roc_auc_score(y_train, proba),
        "pr_auc": average_precision_score(y_train, proba),
        f"at_threshold_{THRESHOLD}": {
            "precision": precision_score(y_train, pred, zero_division=0),
            "recall": recall_score(y_train, pred),
            "confusion_matrix": confusion_matrix(y_train, pred).tolist(),
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    click.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
