#!/bin/bash
# One-command work session: launch -> wait for provisioning -> connect ->
# (you work; type `exit`) -> confirm teardown.
# Usage (from anywhere in the repo):
#   bash scripts/session.sh        -> t3.large, plain shell
#   bash scripts/session.sh -t     -> with Jupyter tunnel
# On exit it ASKS before terminating, so a dropped connection or "back after
# lunch" doesn't kill the box unless you say so.
set -uo pipefail

# Location-independent: find sibling scripts relative to this file, not the cwd.
DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$DIR/launch.sh" || { echo "Launch failed — aborting session."; exit 1; }

# Poll for the provisioning marker instead of sleeping a guessed duration:
# "wait until done", not "wait two minutes and hope".
TAG="cc-fraud"
PEM=~/.ssh/cc-fraud-key.pem
IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Waiting for boot provisioning on $IP ..."
until ssh -i "$PEM" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR -o ConnectTimeout=5 ec2-user@"$IP" 'test -f ~/PROVISIONED' 2>/dev/null; do
  echo "  ...still provisioning (checking again in 15s)"
  sleep 15
done
echo "Provisioned — connecting."

bash "$DIR/connect.sh" "$@" || true   # returns here when the SSH session ends

read -r -p "Session ended. Terminate the instance now? [Y/n] " ans
if [ "${ans:-Y}" != "n" ] && [ "${ans:-Y}" != "N" ]; then
  bash "$DIR/teardown.sh"
else
  echo "Left running — remember: bash scripts/teardown.sh when done. (Meter is on.)"
fi
