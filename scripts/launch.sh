#!/bin/bash
# Launch a provisioned work instance. Usage:
#   bash launch.sh              -> t3.large (default)
#   bash launch.sh g4dn.xlarge  -> GPU box (quota: max one)
# The instance provisions itself at boot (user-data): installs Python/Jupyter,
# clones the repo, pulls train.csv. Ready ~2-3 min after launch.
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash: don't mangle /aws/... args (no-op elsewhere)

TYPE="${1:-t3.large}"
TAG="cc-fraud"
KEY="cc-fraud-key"
SG="cc-fraud-ssh"
PROFILE="cc-fraud-ec2"
REPO="https://github.com/shawnstrausser/CC_Fraud.git"
BUCKET="s3://cc-fraud-381491853558"

# Stateless design: one box at a time. Refuse to double-launch.
RUNNING=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$RUNNING" ]; then
  echo "Already running: $RUNNING  (use connect.sh, or teardown.sh first)"; exit 1
fi

AMI=$(aws ssm get-parameter \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query Parameter.Value --output text)

# Boot-time provisioning script (runs once, as root; log: /var/log/provision.log)
# NOTE: relative path on purpose — Git Bash's /tmp is invisible to aws.exe,
# and MSYS_NO_PATHCONV (needed above) disables the auto-translation.
USERDATA=".userdata.$$.tmp"
trap 'rm -f "$USERDATA"' EXIT
cat > "$USERDATA" <<EOF
#!/bin/bash
exec > /var/log/provision.log 2>&1
set -x
dnf -y install git python3.11 python3.11-pip
sudo -u ec2-user bash -c '
  cd ~
  python3.11 -m pip install --user jupyterlab pandas scikit-learn matplotlib seaborn xgboost
  git clone $REPO
  cd CC_Fraud
  aws s3 cp $BUCKET/data/train.csv .
'
touch /home/ec2-user/PROVISIONED   # marker: setup finished
EOF

ID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type "$TYPE" \
  --key-name "$KEY" --security-groups "$SG" \
  --iam-instance-profile "Name=$PROFILE" \
  --user-data "file://$USERDATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "Launched $ID ($TYPE) — meter is running."

aws ec2 wait instance-running --instance-ids "$ID"
IP=$(aws ec2 describe-instances --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Instance up: $IP"
echo "Provisioning takes ~2-3 min more (marker file ~/PROVISIONED appears when done)."
echo "Next: bash connect.sh        (shell)"
echo "      bash connect.sh -t     (shell + Jupyter tunnel on port 8888)"
