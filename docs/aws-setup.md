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

## 4. EC2 instance (Postgres + MLflow) *(Phase 1)*

On the instance (Docker and the AWS CLI installed, instance role allowed to
read `/leaf-al/*` in SSM and write the models bucket):

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
