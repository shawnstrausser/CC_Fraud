"""Baseline fraud model: logistic regression.

Usage: python train.py <train_csv> <output_dir>
Trains on the frozen train split (data.py) and evaluates on the
TRAINING data — fit quality, not generalization. The frozen test.csv
stays untouched until final model comparison.
Writes metrics.json to <output_dir>.
"""
import json
import sys
import time
from pathlib import Path

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

train_csv, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(train_csv)
X_train, y_train = train.drop(columns=["Class"]), train["Class"]

# Logistic regression needs features on comparable scales to converge.
scaler = StandardScaler().fit(X_train)
model = LogisticRegression(max_iter=1000)
t0 = time.perf_counter()
model.fit(scaler.transform(X_train), y_train)
train_seconds = time.perf_counter() - t0

proba = model.predict_proba(scaler.transform(X_train))[:, 1]
pred = proba >= THRESHOLD

metrics = {
    "model": "logistic_regression_baseline",
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
print(json.dumps(metrics, indent=2))
