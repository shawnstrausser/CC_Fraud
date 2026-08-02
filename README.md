# Credit Card Fraud Detection

Binary classification on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 frauds (0.172%). Trained on AWS EC2; data and results in S3; this repo is the source of truth for all code.

## Layout

| Path | What |
|---|---|
| `fraud.ipynb` | EDA (train split only — see protocol below) |
| `data.py` | One-time frozen train/test split |
| `train.py` | Fit + save the model artifact (`model.joblib` + `train_meta.json`) |
| `evaluate.py` | Score a saved model on any dataset → `metrics.json`; also the one-shot test evaluator |
| `scripts/` | Session automation: `session.sh`, `launch.sh`, `connect.sh`, `teardown.sh` |
| `infra/` | IAM policy documents |
| `docs/runbook.md` | **The AWS runbook**: setup, per-session commands, gotchas |

## Quick start (per session)

```bash
git add -A && git commit && git push     # servers only run committed code
bash scripts/session.sh [-t]             # launch -> provision -> connect; -t = Jupyter tunnel
# ... work ...; type exit -> confirms teardown
```

Details, manual fallbacks, and hard-won gotchas: [docs/runbook.md](docs/runbook.md).

## Evaluation protocol

Frozen 80/20 time-based split (`data.py`, run once; files live in `s3://cc-fraud-381491853558/data/`):

- train: 227,845 rows, 417 frauds (0.183%)
- test: 56,962 rows, 75 frauds (0.132%)

Note the base-rate mismatch between train and test — possibly real non-stationarity, possibly burst noise (~2σ). Consequences: with 75 test frauds, metric differences under ~3 recall points are noise; threshold-based metrics (precision/recall @ 0.5) are base-rate-sensitive and shouldn't be compared across datasets. All models eval on this same frozen test set. TODO: consider walk-forward (TimeSeriesSplit) validation inside the train portion for model tuning.

Baseline (logreg, unweighted): train PR-AUC 0.770, recall@0.5 0.62 (misses 158/417 train frauds). The one initial test eval showed a negligible train–test gap → diagnosis: underfitting, not overfitting. Test metrics are deliberately not recorded here — we iterate against train/validation only and touch test.csv again at final model comparison.

EDA findings (fraud.ipynb): fraud-rate spikes at night while volume is diurnal (burstiness, not smooth drift); fraud Amount is bimodal (~$1 card-testing + ~$120 cash-out); top univariate separators V17, V14, V12, V10, V16, V3 — all "narrow legit spike vs wide left fraud smear"; V-V correlations ≈ 0 (PCA), Amount correlates with V2/V5/V7/V20.

## Backlog

1. Class-weighted logistic regression (`class_weight="balanced"`) — the baseline misses 158/417 train frauds at 0.5 because the loss treats fraud and non-fraud errors equally. Cheapest fix first. Prediction to test: recall@0.5 jumps but PR-AUC barely moves (weighting ≈ intercept shift for linear models).
2. XGBoost — capacity jump; also the model that can use the GPU (`device="cuda"`) on the approved g4dn.xlarge.
3. Batch-job pattern — ✅ built as the SSH-driven orchestrator `scripts/experiment.sh` (one command: launch → run → fetch → teardown, teardown guaranteed by trap). Still open: the fire-and-forget user-data variant (self-terminating instance) for runs that should survive the laptop sleeping.
4. Imbalance-strategy comparison (on train-eval; no validation set — scope decision 2026-08-02): class weighting (✗ rejected — PR-AUC down, NE 8.0, calibration 42×) vs negative downsampling vs plain threshold tuning. Keep a strategy only if it lifts the PR curve, not just the operating point. Note: downsampling changes the base rate → recalibrate probabilities (or correct analytically) before comparing threshold metrics.
5. Report model weights in metrics.json — partially superseded: weights are now inspectable by loading `model.joblib` (`pipe.named_steps["model"].coef_`); still open if we want them printed in metrics.
6. ✅ Persist the model artifact every run — `train.py` saves the sklearn Pipeline (scaler + model) as `model.joblib`, uploaded to `s3://…/results/RUN_NAME/` beside metrics.json. Deployment = download + `joblib.load` + `predict_proba` behind whatever serves it (batch script, Lambda, or an endpoint — decide when we get there).
7. ✅ Provenance — `train_meta.json` records the git commit (embedded into metrics.json by `evaluate.py`); S3 `code/` folder deleted.
8. New features from the EDA (judged on train-eval PR-AUC; final verdict at the one-shot test comparison):
   - Time-of-day bucketized into morning / afternoon / night (one-hot). The EDA showed fraud-rate spikes at night.
   - Hour-of-day as the cyclic pair `hour_sin` + `hour_cos` (keep both as separate features — their *ratio* is tan(), which blows up twice a day at cos=0 and repeats every 12h; the pair encodes the clock cleanly).
   - Transactions-per-hour (activity context — computable at serving time from recent traffic).
   - Fraud-rate-per-hour-of-day — ⚠ uses labels → target-encoding leakage risk: compute the rates on training folds only, never on the row's own fold, and only from past data at serving time.
   - Pairwise products of the EDA shortlist (V17, V14, V12, V10, V16, V3) — 15 interaction features to let the linear model see "jointly moderate" frauds.
9. When notebook history gets heavy: commit `fraud.ipynb` with outputs stripped and save key figures as PNGs in `figures/` (git keeps every historical output blob forever).

(Plus the walk-forward validation TODO above.)

## Setup (local)

Download `creditcard.csv` from Kaggle into this folder (gitignored — ~150MB). Needs: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`.
