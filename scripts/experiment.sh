#!/bin/bash
# One-command experiment: verify git state -> launch -> wait -> run remotely ->
# fetch results -> teardown (teardown happens even if the run fails).
#
# Usage:
#   bash scripts/experiment.sh <RUN_NAME> [train.py options...]
#   bash scripts/experiment.sh logreg-baseline
#   bash scripts/experiment.sh logreg-balanced --class-weight balanced
#   bash scripts/experiment.sh eda-refresh --notebook     (re-executes fraud.ipynb instead)
#
# Results land in results/<RUN_NAME>/ locally AND s3://.../results/<RUN_NAME>/.
set -euo pipefail
export MSYS_NO_PATHCONV=1

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$DIR/.." && pwd)"
TAG="cc-fraud"
PEM=~/.ssh/cc-fraud-key.pem
BUCKET="s3://cc-fraud-381491853558"
SSH_OPTS=(-i "$PEM" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

RUN_NAME="${1:?usage: experiment.sh <RUN_NAME> [--script foo.py] [--notebook] [script options...]}"
shift
NOTEBOOK=0
SCRIPT="train.py"
while [ $# -gt 0 ]; do
  case "$1" in
    --notebook) NOTEBOOK=1; shift ;;
    --script)   SCRIPT="$2"; shift 2 ;;
    *) break ;;
  esac
done
TRAIN_ARGS=("$@")

# --- Safety: servers only run committed, pushed code -------------------------
cd "$REPO_DIR"
if [ -n "$(git status --porcelain)" ]; then
  echo "ABORT: uncommitted changes. Commit (and push) first."; exit 1
fi
if [ "$(git rev-list origin/main..main --count)" -gt 0 ]; then
  echo "ABORT: local commits not pushed. Run: git push"; exit 1
fi

# --- Launch and wait for provisioning ---------------------------------------
bash "$DIR/launch.sh"
trap 'bash "$DIR/teardown.sh"' EXIT   # meter off no matter how we exit from here on

IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Waiting for provisioning on $IP ..."
until ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 ec2-user@"$IP" 'test -f ~/PROVISIONED' 2>/dev/null; do
  echo "  ...still provisioning"; sleep 15
done

# --- Run remotely ------------------------------------------------------------
if [ "$NOTEBOOK" -eq 1 ]; then
  echo "Re-executing fraud.ipynb remotely..."
  ssh "${SSH_OPTS[@]}" ec2-user@"$IP" \
    "cd CC_Fraud && python3.11 -m jupyter nbconvert --to notebook --execute --inplace fraud.ipynb"
  scp "${SSH_OPTS[@]}" ec2-user@"$IP":CC_Fraud/fraud.ipynb "$REPO_DIR/fraud.ipynb"
  echo "Notebook fetched. Review, then commit+push it yourself."
else
  echo "Run '$RUN_NAME' ($SCRIPT) with args: ${TRAIN_ARGS[*]:-(none)}"
  if [ "$SCRIPT" = "train.py" ]; then
    # Default: train (saves model.joblib) then evaluate it on the train data.
    REMOTE="python3.11 train.py train.csv out/ ${TRAIN_ARGS[*]:-} \
      && python3.11 evaluate.py out/model.joblib train.csv out/"
  else
    REMOTE="python3.11 $SCRIPT train.csv out/ ${TRAIN_ARGS[*]:-}"
  fi
  ssh "${SSH_OPTS[@]}" ec2-user@"$IP" \
    "cd CC_Fraud && $REMOTE && aws s3 cp out/ $BUCKET/results/$RUN_NAME/ --recursive"
  mkdir -p "$REPO_DIR/results/$RUN_NAME"
  scp -r "${SSH_OPTS[@]}" ec2-user@"$IP":"CC_Fraud/out/*" "$REPO_DIR/results/$RUN_NAME/"
  echo "--- fetched into results/$RUN_NAME/:"
  ls "$REPO_DIR/results/$RUN_NAME"
  if [ -f "$REPO_DIR/results/$RUN_NAME/metrics.json" ]; then
    cat "$REPO_DIR/results/$RUN_NAME/metrics.json"
  fi
fi
# trap fires here -> teardown + paranoia sweep
