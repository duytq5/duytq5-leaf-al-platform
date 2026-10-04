# AWS setup (run by the owner)

Everything in this repo builds and tests without AWS credentials. Nothing here
is deployed automatically; you run these commands yourself, from your own
machine with your own AWS profile. Sections marked *(Phase 1)* need resources
the CDK stack does not create yet; they become real as Phase 1 lands.

Set these once per shell:

```bash
export AWS_PROFILE=<your-profile>
export AWS_REGION=ap-southeast-1          # pick one region and keep it
export CDK_DEFAULT_REGION=$AWS_REGION
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
```

## 1. Budget alert (do this first)

```bash
aws budgets create-budget --account-id "$CDK_DEFAULT_ACCOUNT" \
  --budget '{"BudgetName":"leaf-al-monthly","BudgetLimit":{"Amount":"20","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80,"ThresholdType":"PERCENTAGE"},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"<your-email>"}]}]'
```

Change `20` to your monthly limit.

## 2. Secrets in SSM Parameter Store

Never commit these. Use URL-safe characters for the Postgres password.

```bash
aws ssm put-parameter --name /leaf-al/postgres/password --type SecureString \
  --value "$(openssl rand -hex 24)"
```

*(Phase 1)* after the stack is deployed, store the bucket names it prints:

```bash
aws ssm put-parameter --name /leaf-al/buckets/data   --type String --value <DataBucketName output>
aws ssm put-parameter --name /leaf-al/buckets/models --type String --value <ModelsBucketName output>
```

## 3. CDK stack

```bash
uv sync --extra infra
source .venv/bin/activate
npx aws-cdk bootstrap "aws://$CDK_DEFAULT_ACCOUNT/$AWS_REGION"   # once per account/region
npx aws-cdk diff                                                 # review what will change
npx aws-cdk deploy LeafAlPlatform
```

To remove everything later: `npx aws-cdk destroy LeafAlPlatform`.

## 4. Persistent data volume *(Phase 1)*

All Postgres data (metadata and the MLflow database, later Label Studio) lives
on a separate EBS volume mounted at `/data`. It is not the instance's root
disk, so it survives the instance being stopped, terminated or replaced.
MLflow artifacts are already in S3. In Phase 1 the CDK stack will create this
volume with `RemovalPolicy.RETAIN`; until then, create it by hand.

The volume costs storage even when no instance is running (gp3 20 GB is about
1.6 USD/month), so it counts toward the budget alongside EC2.

### Create and attach (once)

The volume must be in the same availability zone as the instance.

```bash
AZ=$(aws ec2 describe-instances --instance-ids <instance-id> \
  --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text)
VOL=$(aws ec2 create-volume --availability-zone "$AZ" --size 20 --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=leaf-al-data}]' \
  --query VolumeId --output text)
aws ec2 wait volume-available --volume-ids "$VOL"
aws ec2 attach-volume --volume-id "$VOL" --instance-id <instance-id> --device /dev/sdf
echo "$VOL"   # keep this id; you need it to re-attach
```

A volume attached this way has DeleteOnTermination off, so terminating the
instance leaves it alone. Check with:

```bash
aws ec2 describe-instances --instance-ids <instance-id> \
  --query 'Reservations[0].Instances[0].BlockDeviceMappings[?DeviceName==`/dev/sdf`].Ebs.DeleteOnTermination'
```

### Format and mount (on the instance)

On Nitro instances the volume shows up as `/dev/nvme1n1`; confirm with `lsblk`.
**Only format a brand-new volume.** `sudo file -s /dev/nvme1n1` prints `data`
when it is empty; if it prints a filesystem, skip `mkfs` or you erase everything.

```bash
lsblk
sudo file -s /dev/nvme1n1
sudo mkfs.ext4 -L leaf-al-data /dev/nvme1n1        # first time only
sudo mkdir -p /data
echo 'LABEL=leaf-al-data /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a && df -h /data
```

`nofail` lets the instance boot even if the volume is missing.

### Moving to a new instance

```bash
aws ec2 attach-volume --volume-id <vol-id> --instance-id <new-instance-id> --device /dev/sdf
```

Then on the new instance run the fstab and mount lines above (no `mkfs`), and
start the services as in section 5. Postgres finds its existing data in
`/data/postgres`.

### Backups

Snapshot the volume before risky changes:

```bash
aws ec2 create-snapshot --volume-id <vol-id> --description "leaf-al before <change>"
```

And dump Postgres to S3 nightly from the instance (`crontab -e`):

```
0 3 * * * /home/ec2-user/<repo>/deploy/ec2/backup-postgres.sh >> /var/log/leaf-al-backup.log 2>&1
```

Restore a dump into a fresh, empty Postgres:

```bash
aws s3 cp s3://<data-bucket>/backups/postgres/<file>.sql.gz - | gunzip \
  | docker compose exec -T postgres psql -U al -d postgres
```

## 5. EC2 services (Postgres + MLflow) *(Phase 1)*

On the instance, with `/data` mounted (section 4), Docker and the AWS CLI
installed, and an instance role allowed to read `/leaf-al/*` in SSM, write the
models bucket, and write `backups/` in the data bucket:

```bash
git clone <this repo> && cd <repo>/deploy/ec2
AWS_REGION=<region> ./write-env.sh     # writes .env from SSM, mode 600
docker compose up -d --build
```

Restrict the security group: 5432 and 5000 only from your IP (and the
Lambdas' needs, decided in Phase 1). Stop the instance when not in use:

```bash
aws ec2 stop-instances --instance-ids <id>
```
