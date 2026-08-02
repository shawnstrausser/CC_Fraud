#!/bin/bash
# SSH into the running cc-fraud instance (found by tag — no state files).
# Usage:
#   bash connect.sh       -> plain shell
#   bash connect.sh -t    -> shell + Jupyter tunnel (browser: localhost:8888)
set -euo pipefail
export MSYS_NO_PATHCONV=1

TAG="cc-fraud"
PEM=~/.ssh/cc-fraud-key.pem

IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
if [ -z "$IP" ] || [ "$IP" = "None" ]; then
  echo "No running $TAG instance. Launch one:  bash launch.sh"; exit 1
fi

TUNNEL=()
if [ "${1:-}" = "-t" ]; then
  TUNNEL=(-L 8888:localhost:8888)
  echo "Tunnel active: Jupyter will be at http://localhost:8888 (keep this window open)"
fi

# Ephemeral servers: IPs get reused, so don't record/verify host fingerprints
# (they'd collide across instances). Fine for throwaway boxes; never do this
# for long-lived servers you care about.
exec ssh -i "$PEM" \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
  "${TUNNEL[@]}" ec2-user@"$IP"
