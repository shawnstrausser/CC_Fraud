#!/bin/bash
# Terminate the cc-fraud instance(s) and verify the account is quiet.
# Usage: bash teardown.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1

TAG="cc-fraud"

IDS=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -z "$IDS" ]; then
  echo "Nothing tagged $TAG is running."
else
  echo "Terminating: $IDS"
  # shellcheck disable=SC2086  # word-splitting IDs is intended
  aws ec2 terminate-instances --instance-ids $IDS --query 'TerminatingInstances[].CurrentState.Name'
  aws ec2 wait instance-terminated --instance-ids $IDS
  echo "Confirmed terminated."
fi

echo "Paranoia sweep (anything still billing, tagged or not):"
aws ec2 describe-instances \
  --filters Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType]'
echo "[] means the meter is fully off."
