# Credit Card Fraud Detection

Binary classification on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 frauds (0.172%).

## Setup

Download `creditcard.csv` from Kaggle and drop it in this folder (gitignored — ~150MB).

Needs: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`.

## Evaluation protocol

Frozen 80/20 time-based split (`data.py`, run once; files live in `s3://cc-fraud-381491853558/data/`):

- train: 227,845 rows, 417 frauds (0.183%)
- test: 56,962 rows, 75 frauds (0.132%)

Note the base-rate mismatch between train and test — possibly real non-stationarity, possibly burst noise (~2σ). Consequences: with 75 test frauds, metric differences under ~3 recall points are noise; threshold-based metrics (precision/recall @ 0.5) are base-rate-sensitive and shouldn't be compared across datasets. All models eval on this same frozen test set. TODO: consider walk-forward (TimeSeriesSplit) validation inside the train portion for model tuning.

Baseline (logreg, unweighted): train PR-AUC 0.770, recall@0.5 0.62 (misses 158/417 train frauds). The one initial test eval showed a negligible train–test gap → diagnosis: underfitting, not overfitting. Test metrics are deliberately not recorded here — we iterate against train/validation only and touch test.csv again at final model comparison.

## Backlog

1. Class-weighted logistic regression (`class_weight="balanced"`) — the baseline misses 158/417 train frauds at 0.5 because the loss treats fraud and non-fraud errors equally. Cheapest fix first. Prediction to test: recall@0.5 jumps but PR-AUC barely moves (weighting ≈ intercept shift for linear models).
2. XGBoost — capacity jump; also the model that can use the GPU (`device="cuda"`) on the approved g4dn.xlarge.
3. Batch-job pattern — rerun training as a self-terminating instance: user-data boot script + `--instance-initiated-shutdown-behavior terminate`.
4. Imbalance-strategy comparison (on a validation slice, not test): class weighting vs negative downsampling (drop a large fraction of non-fraud rows from training) vs plain threshold tuning. Judge by validation PR-AUC — keep a strategy only if it lifts the curve, not just the operating point. Note: downsampling changes the base rate → recalibrate probabilities (or correct analytically) before comparing threshold metrics.
5. Report model weights in metrics.json — for logreg, the 30 coefficients (on scaled features they double as rough feature importances) + intercept.
6. Persist the model artifact itself every run so it can be deployed (currently only metrics are saved — the baseline model died with the server). `joblib.dump` the sklearn pipeline (scaler + model together — coefficients are meaningless without the scaler's train-set means/stds), upload to `s3://…/results/RUN_NAME/model.joblib` beside metrics.json. Deployment then = download artifact + `joblib.load` + `predict_proba` behind whatever serves it (batch script, Lambda, or an endpoint — decide when we get there).
7. Record the git commit hash in metrics.json (`git rev-parse --short HEAD`) so every result links to the exact code that produced it. Delete the now-retired `code/` folder from S3.

(Plus the walk-forward validation TODO above.)

## AWS runbook

Account `381491853558`, region `us-east-1`, bucket `s3://cc-fraud-381491853558`. Commands are bash (Git Bash).

### One-time setup (done — recorded for reference)

```bash
# Identity & CLI: AWS account -> MFA on root -> IAM user shawn-admin
# (AdministratorAccess) -> access keys -> aws configure. Budget alert in console.
aws sts get-caller-identity                      # verify CLI is connected

# Storage
aws s3 mb s3://cc-fraud-381491853558
aws s3 cp creditcard.csv s3://cc-fraud-381491853558/data/creditcard.csv

# SSH key pair (private half saved ONCE; lives outside the repo)
aws ec2 create-key-pair --key-name cc-fraud-key --query KeyMaterial --output text > ~/.ssh/cc-fraud-key.pem

# Firewall: SSH only, from my IPv4 only
curl -4 ifconfig.me
aws ec2 create-security-group --group-name cc-fraud-ssh --description "SSH from my IP only"
aws ec2 authorize-security-group-ingress --group-name cc-fraud-ssh --protocol tcp --port 22 --cidr MY_IPV4/32

# IAM role so instances reach S3 with no stored keys (trust.json / s3-policy.json in repo)
aws iam create-role --role-name cc-fraud-ec2 --assume-role-policy-document file://trust.json
aws iam put-role-policy --role-name cc-fraud-ec2 --policy-name s3-bucket-access --policy-document file://s3-policy.json
aws iam create-instance-profile --instance-profile-name cc-fraud-ec2
aws iam add-role-to-instance-profile --instance-profile-name cc-fraud-ec2 --role-name cc-fraud-ec2

# GPU quota (approved): Service Quotas -> EC2 -> L-DB2E81BA "Running On-Demand G and VT instances" = 4 vCPUs
```

### Per training session

```bash
# 1. Launch (meter starts, t3.large ~8c/hr)
MSYS_NO_PATHCONV=1 aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text
aws ec2 run-instances --image-id AMI_ID --instance-type t3.large \
  --key-name cc-fraud-key --security-groups cc-fraud-ssh \
  --iam-instance-profile Name=cc-fraud-ec2 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cc-fraud-ssh-train}]' \
  --query 'Instances[0].InstanceId' --output text
aws ec2 describe-instances --instance-ids INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# 2. Code ships via git (decided 2026-08-01): commit + push BEFORE the run;
#    servers only ever run committed code. S3 carries data/artifacts only.
git add -A && git commit && git push

# 3. On the server
ssh -i ~/.ssh/cc-fraud-key.pem ec2-user@PUBLIC_IP
sudo dnf -y install git python3.11 python3.11-pip
python3.11 -m pip install pandas scikit-learn
git clone https://github.com/shawnstrausser/CC_Fraud.git && cd CC_Fraud
#   (private repo: create a fine-grained PAT — repo-scoped, Contents: read-only —
#    and clone with https://TOKEN@github.com/shawnstrausser/CC_Fraud.git)
aws s3 cp s3://cc-fraud-381491853558/data/train.csv split/train.csv
python3.11 train.py split/train.csv out_train/
aws s3 cp out_train/metrics.json s3://cc-fraud-381491853558/results/RUN_NAME/metrics.json

# 4. Terminate and VERIFY (meter stops)
aws ec2 terminate-instances --instance-ids INSTANCE_ID
aws ec2 describe-instances --instance-ids INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text
```

### Gotchas learned the hard way

- Git Bash rewrites args starting with `/` into Windows paths — prefix `MSYS_NO_PATHCONV=1`.
- PowerShell redirects (`>`) corrupt key files with encoding bytes — in PS use `Out-File -Encoding ascii`; in Git Bash plain `>` is fine.
- `ifconfig.me` may return IPv6; security groups' `--cidr` wants IPv4 — use `curl -4`.
- The private key is shown exactly once at create-key-pair; a broken pipe there means delete the pair and recreate.
- S3 `cp` to an existing key overwrites silently — that's how results/code get updated.

Everything lives in `fraud.ipynb`.
