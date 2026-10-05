# AWS setup (run by the owner)

Everything in this repo builds and tests without AWS credentials. Nothing here
is deployed automatically; you run these commands yourself, from your own
machine with your own AWS profile. Sections marked *(later in Phase 1)* need
resources the CDK stack does not create yet.

Set these once per shell:

```bash
export AWS_PROFILE=<your-profile>
export AWS_REGION=ap-southeast-1          # pick one region and keep it
export CDK_DEFAULT_REGION=$AWS_REGION
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export MY_IP=$(curl -s https://checkip.amazonaws.com)/32
```

## What the stack creates

`LeafAlPlatform` (`infra/stacks/platform.py`):

| Resource | Notes |
| --- | --- |
| S3 data bucket (`al-data`) | Private, SSE-S3, HTTPS only. `backups/` objects expire after 30 days. Kept on `cdk destroy`. |
| S3 models bucket (`al-models`) | Same, plus versioning (old versions expire after 90 days). Kept on `cdk destroy`. |
| SQS job queue + dead-letter queue | 20 s long polling, 30 min visibility, 14-day retention, a job goes to the DLQ after 3 failed receives. |
| IAM user for the GPU worker | Consume jobs, report Step Functions task results, read `raw/` and `manifests/`, write `scores/`, read/write `mlflow-artifacts/`. No access key: you create it (section 4). |
| VPC + EC2 instance | One public subnet, no NAT gateway. Amazon Linux 2023, `t3.small`, IMDSv2, no SSH key: connect with Session Manager. Docker, compose and cron installed on first boot. |
| EBS data volume (`/data`) | gp3 20 GB, encrypted, mounted at `/data` on first boot. Kept when the instance or the stack is deleted. |
| Security group | MLflow (5000) and Postgres (5432) open only to `adminCidr` and `workerCidr`. Nothing else inbound. |
| SSM parameters | `/leaf-al/buckets/data`, `/leaf-al/buckets/models`, `/leaf-al/queues/jobs-url`. |
| AWS Budget | Monthly cost budget, email at 80 % actual and 100 % forecast. Only created when `budgetEmail` is set. |

Deploy-time settings are CDK context values (`-c key=value`):

| Key | Default | Meaning |
| --- | --- | --- |
| `adminCidr` | none (no inbound access) | Your IP as `/32` |
| `workerCidr` | none | The GPU worker's IP, if it is not the same as `adminCidr` |
| `budgetEmail` | none (no Budget) | Where Budget alerts go |
| `budgetUsd` | `20` | Monthly budget in USD |
| `instanceType` | `t3.small` | EC2 instance type (x86_64) |
| `dataVolumeGb` | `20` | Size of the `/data` volume |

## 1. Secrets in SSM Parameter Store (before the first deploy)

CloudFormation cannot create SecureString parameters, so the Postgres password
is yours to create. Never commit it. Use URL-safe characters (it goes into the
MLflow backend URI):

```bash
aws ssm put-parameter --name /leaf-al/postgres/password --type SecureString \
  --value "$(openssl rand -hex 24)"
```

The stack writes the bucket names and queue URL to SSM itself.

## 2. CDK stack

```bash
uv sync --extra infra
source .venv/bin/activate
npx aws-cdk bootstrap "aws://$CDK_DEFAULT_ACCOUNT/$AWS_REGION"   # once per account/region

CTX="-c adminCidr=$MY_IP -c budgetEmail=<your-email> -c budgetUsd=20"
npx aws-cdk diff   LeafAlPlatform $CTX                           # review what will change
npx aws-cdk deploy LeafAlPlatform $CTX
```

Use the same `-c` values on every deploy. A deploy without `adminCidr` closes
MLflow and Postgres again; a deploy without `budgetEmail` deletes the Budget.
If your home IP changes, re-run the deploy with the new `MY_IP`.

If you already created a budget named `leaf-al-monthly` by hand (the earlier
version of this doc did), delete it first, or the deploy fails on the name:
`aws budgets delete-budget --account-id "$CDK_DEFAULT_ACCOUNT" --budget-name leaf-al-monthly`.

AWS sends a confirmation email for the Budget alert subscription; confirm it.

The deploy prints these outputs; you need them below:
`DataBucketName`, `ModelsBucketName`, `JobQueueUrl`, `JobDlqUrl`,
`WorkerUserName`, `InstanceId`, `DataVolumeId`.

```bash
aws cloudformation describe-stacks --stack-name LeafAlPlatform \
  --query 'Stacks[0].Outputs' --output table
```

Before deploying a change, read the `diff`. If it says the `Host` instance
will be **replaced**, stop and ask first: the data volume has to be detached
from the old instance before the new one can attach it.

### Removing the stack

```bash
npx aws-cdk destroy LeafAlPlatform
```

This deletes the instance, queues, IAM user and parameters, but **keeps** both
buckets and the data volume, so no data is lost by accident. They keep costing
storage until you delete them by hand (S3 console, or `aws s3 rb --force` and
`aws ec2 delete-volume --volume-id <DataVolumeId>`). A later deploy creates a
new, empty volume and new buckets; restore Postgres from a backup (section 5).

## 3. EC2 services (Postgres + MLflow)

Connect with Session Manager (install the
[Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
for the AWS CLI once):

```bash
aws ssm start-session --target <InstanceId>
```

On the instance, check that first-boot setup finished and `/data` is mounted,
then start the services:

```bash
sudo tail -n 20 /var/log/cloud-init-output.log
df -h /data
sudo su - ec2-user
git clone https://github.com/duytq5/duytq5-leaf-al-platform.git && cd duytq5-leaf-al-platform/deploy/ec2
AWS_REGION=<region> ./write-env.sh     # writes .env from SSM, mode 600
docker compose up -d --build
```

The instance's public IP changes every time it starts (no Elastic IP, to save
its hourly charge). Get the current one with:

```bash
aws ec2 describe-instances --instance-ids <InstanceId> \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

MLflow is then at `http://<public-ip>:5000` from your IP.

Stop the instance when not in use (the volume keeps all data):

```bash
aws ec2 stop-instances  --instance-ids <InstanceId>
aws ec2 start-instances --instance-ids <InstanceId>
```

Postgres opens to the Lambdas (which run outside a VPC) only once it has TLS
and a strong password, in the database schema PR.

## 4. Worker credentials (on the RTX 5070 machine)

The stack creates the worker's IAM user but not its access key, so the key
never appears in CloudFormation. Create it and store it only on the GPU box:

```bash
aws iam create-access-key --user-name <WorkerUserName>
# on the GPU machine:
aws configure --profile leaf-al-worker     # paste the key id and secret, same region
```

To rotate: create a second key, switch the worker to it, then
`aws iam delete-access-key --user-name <WorkerUserName> --access-key-id <old-id>`.

If the worker runs from a different IP than you, add `-c workerCidr=<its-ip>/32`
to the deploy so it can reach MLflow.

## 5. Persistent data volume

All Postgres data (metadata and the MLflow database, later Label Studio) lives
on the separate EBS volume mounted at `/data`. MLflow artifacts are in S3.
First boot formats the volume only if it has no filesystem yet, labels it
`leaf-al-data`, and adds it to `/etc/fstab` with `nofail`, so stopping,
starting and rebooting keep everything.

The volume costs storage even while the instance is stopped (gp3 20 GB is
about 1.6 USD/month), so it counts toward the budget alongside EC2.

### Backups

Snapshot the volume before risky changes:

```bash
aws ec2 create-snapshot --volume-id <DataVolumeId> --description "leaf-al before <change>"
```

And dump Postgres to S3 nightly from the instance (`crontab -e` as `ec2-user`):

```
0 3 * * * /home/ec2-user/duytq5-leaf-al-platform/deploy/ec2/backup-postgres.sh >> /home/ec2-user/leaf-al-backup.log 2>&1
```

Dumps are kept for 30 days. Restore one into a fresh, empty Postgres:

```bash
aws s3 cp s3://<DataBucketName>/backups/postgres/<file>.sql.gz - | gunzip \
  | docker compose exec -T postgres psql -U al -d postgres
```

## 6. Later in Phase 1

The `al-round` state machine, the selection, oracle, snapshot and merge
Lambdas, and Cognito are added to the same stack by later PRs; this file gets
their steps as they land.
