# AWS runbook

Account `381491853558`, region `us-east-1`, bucket `s3://cc-fraud-381491853558`. Commands are bash (Git Bash). Placeholders in `<ANGLE_BRACKETS>`; everything else runs as-is.

## Per session (scripted — the normal path)

```bash
bash scripts/session.sh [-t]      # launch -> wait -> connect -> confirm teardown
```

Or à la carte:

```bash
bash scripts/launch.sh [<INSTANCE_TYPE>]   # default t3.large; g4dn.xlarge = GPU box
bash scripts/connect.sh [-t]               # -t adds the Jupyter tunnel (localhost:8888)
bash scripts/teardown.sh                   # terminate + verify + paranoia sweep
```

The launch provisions the box at boot (Python/Jupyter, repo clone, train.csv; marker file `~/PROVISIONED`, log `/var/log/provision.log` on the box). Jupyter still starts manually on the server: `python3.11 -m jupyter lab --no-browser`.

Code ships via git: commit + push BEFORE the run; servers only ever run committed code. S3 carries data/artifacts only.

## One-time setup (done — recorded for reference)

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
aws ec2 authorize-security-group-ingress --group-name cc-fraud-ssh --protocol tcp --port 22 --cidr <MY_IPV4>/32

# IAM role so instances reach S3 with no stored keys (policy docs in infra/)
aws iam create-role --role-name cc-fraud-ec2 --assume-role-policy-document file://infra/trust.json
aws iam put-role-policy --role-name cc-fraud-ec2 --policy-name s3-bucket-access --policy-document file://infra/s3-policy.json
aws iam create-instance-profile --instance-profile-name cc-fraud-ec2
aws iam add-role-to-instance-profile --instance-profile-name cc-fraud-ec2 --role-name cc-fraud-ec2

# GPU quota (approved): Service Quotas -> EC2 -> L-DB2E81BA "Running On-Demand G and VT instances" = 4 vCPUs
```

## Manual per-session reference (what the scripts automate)

```bash
# 1. Launch (meter starts, t3.large ~8c/hr)
MSYS_NO_PATHCONV=1 aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text
aws ec2 run-instances --image-id <AMI_ID> --instance-type t3.large \
  --key-name cc-fraud-key --security-groups cc-fraud-ssh \
  --iam-instance-profile Name=cc-fraud-ec2 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cc-fraud}]' \
  --query 'Instances[0].InstanceId' --output text
aws ec2 describe-instances --instance-ids <INSTANCE_ID> --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# 2. On the server
ssh -i ~/.ssh/cc-fraud-key.pem ec2-user@<PUBLIC_IP>
sudo dnf -y install git python3.11 python3.11-pip
python3.11 -m pip install pandas scikit-learn
git clone https://github.com/shawnstrausser/CC_Fraud.git && cd CC_Fraud
aws s3 cp s3://cc-fraud-381491853558/data/train.csv .
python3.11 train.py train.csv out_train/
aws s3 cp out_train/metrics.json s3://cc-fraud-381491853558/results/<RUN_NAME>/metrics.json

# 3. Terminate and VERIFY (meter stops)
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
aws ec2 describe-instances --instance-ids <INSTANCE_ID> --query 'Reservations[0].Instances[0].State.Name' --output text
aws ec2 describe-instances --filters Name=instance-state-name,Values=running,pending --query 'Reservations[].Instances[].[InstanceId,InstanceType]'   # paranoia sweep: [] = quiet
```

## Remote Jupyter (tunnel)

`bash scripts/connect.sh -t`, then on the server `python3.11 -m jupyter lab --no-browser`, then open the printed token URL in the laptop browser. The notebook file saves on the SERVER — commit/push or `scp` it back before teardown:

```bash
scp -i ~/.ssh/cc-fraud-key.pem ec2-user@<PUBLIC_IP>:CC_Fraud/EDA.ipynb .
```

## Gotchas learned the hard way

- Git Bash rewrites args starting with `/` into Windows paths — prefix `MSYS_NO_PATHCONV=1`.
- ...but Git Bash's `/tmp` is invisible to Windows programs (`aws.exe`), and `MSYS_NO_PATHCONV` disables the auto-translation that would fix it — so scripts that hand files to `aws` should use **relative paths** (`.userdata.tmp`), which mean the same thing in both worlds.
- PowerShell redirects (`>`) corrupt key files with encoding bytes — in PS use `Out-File -Encoding ascii`; in Git Bash plain `>` is fine.
- `ifconfig.me` may return IPv6; security groups' `--cidr` wants IPv4 — use `curl -4`.
- The private key is shown exactly once at create-key-pair; a broken pipe there means delete the pair and recreate.
- S3 `cp` to an existing key overwrites silently — that's how results/code get updated.
- A `git clone` prompting for a username usually means a mistyped URL or a private repo — never type your password; use a PAT (or make the repo public).
