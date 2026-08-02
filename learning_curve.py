"""Learning curve: metrics vs number of fraud examples in training.

Usage:
    python learning_curve.py <train_csv> <out_dir> [--counts 25,50,100,200,all] [--seeds 5]

Sub-splits the (already frozen) training data by time: fit on the first 80%,
evaluate EVERY curve point on the same untouched last 20% (validation slice).
The frozen test.csv is never touched. For each fraud-count N, the fit set is
all legit rows + N randomly subsampled frauds, repeated over several seeds.
Writes learning_curve.json and learning_curve.png to <out_dir>.
"""
import json
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")  # headless server: render to file, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

C_LINE = "#0072B2"


def fit_and_eval(fit_df, val_X, val_y):
    X, y = fit_df.drop(columns=["Class"]), fit_df["Class"]
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)
    proba = model.predict_proba(scaler.transform(val_X))[:, 1]
    base = val_y.mean()
    base_entropy = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    return {
        "roc_auc": roc_auc_score(val_y, proba),
        "pr_auc": average_precision_score(val_y, proba),
        "normalized_entropy": log_loss(val_y, proba) / base_entropy,
    }


@click.command(help=__doc__)
@click.argument("train_csv", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(path_type=Path))
@click.option("--counts", default="25,50,100,200,all", show_default=True,
              help="Comma-separated fraud counts; 'all' = every fraud in the fit pool.")
@click.option("--seeds", default=5, show_default=True,
              help="Random subsamples per count (mean + min-max band).")
@click.option("--val-frac", default=0.2, show_default=True,
              help="Fraction of the train period held out (by time) as the fixed eval slice.")
def main(train_csv, out_dir, counts, seeds, val_frac):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv).sort_values("Time")
    cut = int(len(df) * (1 - val_frac))
    fit_pool, val = df.iloc[:cut], df.iloc[cut:]
    val_X, val_y = val.drop(columns=["Class"]), val["Class"]
    frauds = fit_pool[fit_pool["Class"] == 1]
    legit = fit_pool[fit_pool["Class"] == 0]
    click.echo(f"fit pool: {len(fit_pool):,} rows / {len(frauds)} frauds | "
               f"val slice: {len(val):,} rows / {int(val_y.sum())} frauds")

    grid = sorted({len(frauds) if c.strip() == "all" else min(int(c), len(frauds))
                   for c in counts.split(",")})
    records = []
    for n in grid:
        for seed in range(seeds):
            sample = frauds.sample(n=n, random_state=seed)
            m = fit_and_eval(pd.concat([legit, sample]), val_X, val_y)
            records.append({"n_frauds": n, "seed": seed, **m})
        click.echo(f"n={n}: done ({seeds} seeds)")

    (out_dir / "learning_curve.json").write_text(json.dumps(records, indent=2))

    res = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for metric, ax in zip(["pr_auc", "roc_auc", "normalized_entropy"], axes):
        g = res.groupby("n_frauds")[metric]
        ax.fill_between(g.mean().index, g.min(), g.max(),
                        color=C_LINE, alpha=0.2, label="min-max over seeds")
        ax.plot(g.mean().index, g.mean(), "o-", color=C_LINE, lw=2, label="mean")
        ax.set(xlabel="frauds in training data", title=metric)
        ax.set_xscale("log")
    axes[0].legend()
    fig.suptitle("Learning curve over positive examples (unweighted logreg, fixed validation slice)")
    fig.savefig(out_dir / "learning_curve.png", dpi=150)
    click.echo(f"wrote learning_curve.json + learning_curve.png to {out_dir}/")


if __name__ == "__main__":
    main()
