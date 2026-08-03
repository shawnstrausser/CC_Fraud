"""Evaluate a saved model artifact on a dataset.

Usage:
    python evaluate.py <model_joblib> <csv> <out_dir> [--threshold 0.5]

Loads the pipeline artifact, scores the dataset, and writes eval_metadata.json.
train_metadata.json found beside the model is embedded, so every evaluation record
names its own config and git commit. This one script serves both train-eval
during iteration and the one-shot final test evaluation — the grader can
never drift between practice and the exam.
"""
import json
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@click.command(help=__doc__)
@click.argument("model_joblib", type=click.Path(exists=True, path_type=Path))
@click.argument("csv", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(path_type=Path))
@click.option("--threshold", default=0.5, show_default=True,
              help="Decision threshold for the precision/recall/confusion block.")
def main(model_joblib, csv, out_dir, threshold):
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(model_joblib)
    df = pd.read_csv(csv)
    X, y = df.drop(columns=["Class"]), df["Class"]
    proba = pipeline.predict_proba(X)[:, 1]
    pred = proba >= threshold

    base_rate = y.mean()
    base_entropy = -(base_rate * np.log(base_rate)
                     + (1 - base_rate) * np.log(1 - base_rate))

    meta_path = model_joblib.parent / "train_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    metrics = {
        **meta,
        "eval_csv": str(csv),
        "eval_rows": len(df),
        "eval_frauds": int(y.sum()),
        "threshold": threshold,
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "normalized_entropy": float(log_loss(y, proba) / base_entropy),
        # Mean predicted probability over the true base rate: 1.0 = calibrated
        # on average, >1 = overpredicts fraud, <1 = underpredicts.
        "calibration_ratio": float(proba.mean() / base_rate),
        f"at_threshold_{threshold}": {
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        },
    }
    (out_dir / "eval_metadata.json").write_text(json.dumps(metrics, indent=2))
    click.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
