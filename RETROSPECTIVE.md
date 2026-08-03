# Retrospective — CC_Fraud

Written 2026-08-03, at project wind-down, as input to the next project
([fraud-at-scale](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF): 21M rows, GPU-scale).
The README holds the technical record; this holds the judgment.

## What this project was

First AWS project and first end-to-end ML project: Kaggle creditcard.csv
(285k rows, 492 frauds), trained on self-provisioned EC2, everything
reproducible from git + S3. Final: logistic regression, 97 engineered
features, **test PR-AUC 0.8007** on an honest time-based split — inside the
0.80–0.88 band that published notebooks reach on easier random splits.
Total cloud spend: ~$2.

## The numbers that tell the story

| Milestone | Train PR-AUC | Meaning |
|---|---|---|
| Baseline logreg (30 cols) | 0.770 | strong start; PCA features carry most signal |
| Class-weighted | 0.734 | rejected — curve down, calibration 42×, NE 8.0 |
| + feature families (97 cols) | 0.855 | error-analysis-guided engineering worked |
| L1 regularization | +0.0004 | at 30× train time — retired |
| XGBoost defaults | 0.998 | memorization demo: train-eval is blind at high capacity |
| **Final test verdict** | **0.8007** | the 5.4-pt gap = measured price of train-eval iteration |

## Keep doing (worked, carry forward)

1. **Frozen splits, materialized as files** — every comparison stayed comparable for the project's whole life.
2. **Artifact pair + provenance** — model.joblib + metadata with git commit; every result self-describing. (MLflow-shaped by hand.)
3. **One-command experiments** (`experiment.sh`): push-guard → launch → run → archive → fetch → guaranteed teardown. Made experiments cheap enough to be honest.
4. **The metric suite** — PR-AUC headline (chosen *before* experiments), NE + calibration ratio alongside. Caught the class-weighting trap ROC-AUC would have hidden.
5. **Error analysis** — the single highest-ROI move: studying the 98 missed frauds produced the gen3 features (+1.3 pts) and the project's best finding (escapees are camouflaged; new *data* beats new transforms).
6. **Registered predictions** before every run, graded after. Kept interpretation honest; the misses taught the most (calibration robustness to drift; L1's irrelevance).
7. **Cattle-not-pets + paranoia sweep** — no forgotten meters; caught two zombie instances from a previous life.
8. **Runbook + gotcha list maintained as they happened** — the cheapest documentation ever written and the most consulted.

## Do differently (paid tuition)

1. **Three-way split from day one.** The no-validation-set scope call was survivable for logreg and fatal for anything bigger: XGBoost pinned train-eval at 0.998 and made tuning impossible, and the final 5.4-pt train→test gap was undiagnosable until spent. At 21M rows there is no excuse: train/val/test, frozen, day one.
2. **Pin dependencies day one** (`requirements.txt` with versions). The sklearn `penalty=` deprecation warning was a free warning shot; unpinned servers get "whatever pip serves today."
3. **Parquet, not CSV** at scale. 1.84GB of CSV is self-harm; columnar + typed from first touch.
4. **One change per experiment when attribution matters.** All-at-once bundles (features+L1 together) repeatedly cost a second run to un-confound. Fine when deliberate; expensive when accidental.
5. **Test automation in the environment that runs it.** Two silent hook failures (cp1252 console, MSYS paths) because scripts were verified in a different shell than they ran in.
6. **Design outputs vs inputs early.** The results-dirty-repo guard loop and the metrics/models reorganizations were all one missing distinction: run *inputs* (code, data refs) vs run *outputs* (results) — decide their homes up front.
7. **Spend the test shot at the end, once.** The early logreg peek was managed with pre-committed rules and honest documentation — but the cleaner story is the one-shot finale. Protocol amendments are debt even when disclosed.
8. **Commit messages are documentation.** `<YOUR_MESSAGE>` shipped three times. Suggested-message-in-command fixed it; start there.

## Open obligations (README backlog 10–11)

- `xgb-default` still owes its pre-committed one-shot test evaluation.
- Deployment-cost evaluation: top-5%-amount segment + $-cost policies
  (FN = amount, FP = $1 parameterized), incl. the per-transaction rule
  `flag iff p × amount > fp_cost` — calibration's payoff, designed, unbuilt.

## Transfer manifest for fraud-at-scale

Copy nearly verbatim: `scripts/` (parameterize bucket/repo/tag at top),
`evaluate.py`, `compare.py`, `.gitattributes`, `.gitignore` hygiene,
`docs/runbook.md` gotchas. Redesign for scale: data loading (Parquet,
maybe polars), split protocol (three-way), instance sizing (GPU earns rent
at 21M rows), experiment tracking (run volume may justify MLflow).
Decided in advance via DESIGN.md, retro in hand, before the first byte
downloads.
