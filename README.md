# Credit Card Fraud Detection

Binary classification on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 frauds (0.172%). Trained on AWS EC2; data and results in S3; this repo is the source of truth for all code.

## Layout

| Path | What |
|---|---|
| `EDA.ipynb` | EDA (train split only — see protocol below) |
| `data.py` | One-time frozen train/test split |
| `train.py` | Fit + save the model artifact (`model.joblib` + `train_metadata.json`) |
| `evaluate.py` | Score a saved model on any dataset → `eval_metadata.json`; also the one-shot test evaluator |
| `compare.py` | Aggregate all results into `results/leaderboard.md` (auto-run by experiment.sh) |
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

## Design decisions

- **Every run's artifact is a pair: model binary + sidecar metadata** (`model.joblib` + `train_metadata.json`). The metadata is the model's *passport* — config, training data, git commit — readable without Python, and it travels with the artifact even if no evaluation ever runs. Evaluations are separate, repeatable events: one model can have many `eval_metadata.json` records (train-eval today, the one-shot test eval later), each embedding a copy of the passport so every record is self-describing. The duplication is deliberate denormalization. This is the standard industry shape — MLflow and friends store exactly this pair, for exactly these reasons. (Decided 2026-08-02.)
- **git for code, S3 for data/artifacts; small human-readable results (JSON/PNG) live in both** — git for visibility next to the code, S3 as system of record. Models and datasets are S3-only (`*.joblib`, `*.csv` gitignored).
- **Notebooks think, scripts produce:** `EDA.ipynb` is for humans forming hypotheses; anything run twice or archived (`train.py`, `evaluate.py`, `learning_curve.py`) is a click-CLI script. Code graduates from notebook to script when it hardens into a procedure.

## Evaluation protocol

Frozen 80/20 time-based split (`data.py`, run once; files live in `s3://cc-fraud-381491853558/data/`):

- train: 227,845 rows, 417 frauds (0.183%)
- test: 56,962 rows, 75 frauds (0.132%)

Note the base-rate mismatch between train and test — possibly real non-stationarity, possibly burst noise (~2σ). Consequences: with 75 test frauds, metric differences under ~3 recall points are noise; threshold-based metrics (precision/recall @ 0.5) are base-rate-sensitive and shouldn't be compared across datasets. All models eval on this same frozen test set. TODO: consider walk-forward (TimeSeriesSplit) validation inside the train portion for model tuning.

Baseline (logreg, unweighted): train PR-AUC 0.770, recall@0.5 0.62 (misses 158/417 train frauds). The one initial test eval showed a negligible train–test gap → diagnosis: underfitting, not overfitting. Test metrics are deliberately not recorded here — we iterate against train/validation only and touch test.csv again at final model comparison.

EDA findings (EDA.ipynb): fraud-rate spikes at night while volume is diurnal (burstiness, not smooth drift); fraud Amount is bimodal (~$1 card-testing + ~$120 cash-out); top univariate separators V17, V14, V12, V10, V16, V3 — all "narrow legit spike vs wide left fraud smear"; V-V correlations ≈ 0 (PCA), Amount correlates with V2/V5/V7/V20.

### Error analysis (EDA.ipynb §8, 2026-08-03)

The champion (`logreg-gen2-l2`, train PR-AUC 0.842) catches signature fraud with near-certainty (median score of caught frauds: 0.988) and misses 98/417 at threshold 0.5. The misses split two ways:

- **~25–30 near-misses** scoring 0.1–0.5 — recoverable by threshold tuning, at a measurable precision cost.
- **~60–70 camouflaged frauds** scoring below 0.01 (missed-group median: 0.0084) — confidently cleared, not narrowly missed. On the strongest separator V17, caught frauds sit at median −9.33 while missed frauds sit at **+0.55** — *indistinguishable from legit traffic (−0.04)*.

Texture: the missed skew toward **daytime** (the caught have a distinct night bump around hours 3–5 — the night features and the extreme-V signature do their work in the dark; what escapes is the fraud that happens at 2pm looking boring) and toward **mid-range amounts** (log-amount 3–6, roughly $20–$400 — not the $1 card-tests, not the signature ~$120 bump, just ordinary-sized purchases).

**Portrait of the escapee:** a mid-sized, daytime transaction whose anonymized features sit squarely inside normal traffic. The model isn't clumsy — these ~60–70 frauds carry almost no signal in the features we have. That's a fundamentally different diagnosis than underfitting, and it bounds what any model can do with this dataset: the escapees justify *new data sources* (merchant category, device fingerprint, per-card history — features that would re-illuminate the invisible), not more transforms of the existing 30 columns. Implications: the threshold sweep has a known prize (~25–30 frauds); XGBoost expectations are tempered (trees can't split on signal that isn't there); published PR-AUC ceilings in the high 0.8s on this dataset are consistent with an irreducible camouflaged remainder.

## Backlog

1. Class-weighted logistic regression (`class_weight="balanced"`) — the baseline misses 158/417 train frauds at 0.5 because the loss treats fraud and non-fraud errors equally. Cheapest fix first. Prediction to test: recall@0.5 jumps but PR-AUC barely moves (weighting ≈ intercept shift for linear models).
2. XGBoost — capacity jump; also the model that can use the GPU (`device="cuda"`) on the approved g4dn.xlarge.
3. Batch-job pattern — ✅ built as the SSH-driven orchestrator `scripts/experiment.sh` (one command: launch → run → fetch → teardown, teardown guaranteed by trap). Still open: the fire-and-forget user-data variant (self-terminating instance) for runs that should survive the laptop sleeping.
4. Imbalance-strategy comparison (on train-eval; no validation set — scope decision 2026-08-02): class weighting (✗ rejected — PR-AUC down, NE 8.0, calibration 42×) vs negative downsampling vs plain threshold tuning. Keep a strategy only if it lifts the PR curve, not just the operating point. Note: downsampling changes the base rate → recalibrate probabilities (or correct analytically) before comparing threshold metrics.
5. Report model weights in eval output — partially superseded: weights are now inspectable by loading `model.joblib` (`pipe.named_steps["model"].coef_`); still open if we want them printed.
6. ✅ Persist the model artifact every run — `train.py` saves the sklearn Pipeline (scaler + model) as `model.joblib`, uploaded to `s3://…/results/RUN_NAME/` beside eval_metadata.json. Deployment = download + `joblib.load` + `predict_proba` behind whatever serves it (batch script, Lambda, or an endpoint — decide when we get there).
7. ✅ Provenance — `train_metadata.json` records the git commit (embedded into eval_metadata.json by `evaluate.py`); S3 `code/` folder deleted.
8. New features from the EDA (judged on train-eval PR-AUC; final verdict at the one-shot test comparison). Implemented in `features.py` behind `train.py --features`, applied INSIDE the saved pipeline (artifact takes raw columns):
   - ✅ `time`: hour_sin/hour_cos cyclic pair + night/morning one-hots (afternoon = reference class).
   - ✅ `interactions`: 15 pairwise products of the EDA shortlist (V17, V14, V12, V10, V16, V3).
   - ✅ `log_amount`: log1p(Amount) — heatmap says partially redundant; testing to confirm.
   - ✅ `activity`: transactions-per-hour (system busyness; label-free).
   - ⏸ Fraud-rate-per-hour-of-day — DEFERRED because it is *target encoding*, which leaks in three subtle ways:
     1. **Self-leak:** computed naively, each row's own label participates in its own feature (a fraud row sees a slightly higher bucket rate *because it's in the bucket* — degenerate case: a 1-row bucket makes the feature literally equal the label). Fix requires out-of-fold encoding: a row's rate comes only from rows in other folds.
     2. **Time-travel leak:** a rate computed over the whole training window gives Monday's rows knowledge of Thursday's frauds; at serving time only past data exists. Fix requires causal (expanding/rolling, strictly-before-t) computation.
     3. **Noise:** 417 frauds ÷ 24 hourly buckets ≈ 17 per bucket — rates are mostly variance; needs smoothing toward the global rate.
     The killer given our protocol: we evaluate on train only, and target-encoded features shine on train *precisely when they leak* — genuine signal and leakage are indistinguishable there. Revisit only with all three fixes built.
9. When notebook history gets heavy: commit `EDA.ipynb` with outputs stripped and save key figures as PNGs in `figures/` (git keeps every historical output blob forever).

(Plus the walk-forward validation TODO above.)

## Setup (local)

Download `creditcard.csv` from Kaggle into this folder (gitignored — ~150MB). Needs: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`.
