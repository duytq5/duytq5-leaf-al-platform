# AWS setup (run by the owner)

Everything in this repo builds and tests without AWS credentials. Nothing here
is deployed automatically; you run these commands yourself, from your own
machine with your own AWS profile. Sections marked *(later in Phase 1)* need
resources the CDK stack does not create yet.

## 0. Local AWS access and the Agent Toolkit (once per machine)

Sign in with `aws login` on your own machine only. Never put AWS credentials in
a cloud session or in this repo. The profile used here is `leaf-al`.

**AWS CLI.** `aws login` and `aws configure agent-toolkit` need a recent AWS
CLI v2 (2.32 is too old for the toolkit). Install or update it:

```bash
curl -fsSL 'https://awscli.amazonaws.com/v2/install.sh' | bash   # installs to ~/.local/bin
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc           # once
hash -r                                                          # or open a new terminal
aws --version
```

On macOS an older install may sit in `/usr/local/bin`. If `aws --version`
still shows the old version, the shell has cached that path: run `hash -r` or
open a new terminal. AWS's `.pkg` installer
(`https://awscli.amazonaws.com/AWSCLIV2.pkg`) updates `/usr/local/bin` instead.

**Sign in and check:**

```bash
aws configure set region ap-southeast-2 --profile leaf-al
aws login --region ap-southeast-2 --profile leaf-al     # browser sign-in
aws sts get-caller-identity --profile leaf-al
```

The sign-in lasts 12 hours and renews for up to 90 days without the browser.

**Agent Toolkit** (AWS skills and the AWS MCP server for Claude Code and other
AI tools). The toolkit service runs only in `us-east-1`, whatever the project
region:

```bash
aws configure agent-toolkit --yes --region us-east-1 --profile leaf-al
aws agent-toolkit list-available-skills --region us-east-1 --profile leaf-al   # check
```

The wizard adds an `aws-mcp` entry to each tool's MCP config (for Claude Code,
`~/.claude.json`). That entry uses the `default` profile, so add this to each
`aws-mcp` entry and restart the tool, or the server fails with
`JSON-RPC error: -32602`:

```json
"env": { "AWS_MCP_PROXY_PROFILES": "leaf-al" }
```

For another account later: `aws login --profile <name>`, add the name to the
space-separated `AWS_MCP_PROXY_PROFILES` list, restart the tool.

| Error | Fix |
| --- | --- |
| `Found invalid choice 'agent-toolkit'` | The old CLI is still being run. `hash -r` or a new terminal, then `aws --version`. |
| `--yes` not recognized | Run the wizard without `--yes`. |
| `Unable to locate credentials` / `ExpiredToken` | Run `aws login --profile leaf-al` again. |
| `command not found: aws` | `export PATH="$HOME/.local/bin:$PATH"` |

The project's rules still apply when an AI tool has AWS access: you deploy, and
nothing creates AWS resources unless you ask for it.

Set these once per shell:

```bash
export AWS_PROFILE=leaf-al
export AWS_REGION=ap-southeast-2          # the project region; keep it
export CDK_DEFAULT_REGION=$AWS_REGION
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
```

## What the stack creates

`LeafAlPlatform` (`infra/stacks/platform.py`):

| Resource | Notes |
| --- | --- |
| S3 data bucket (`al-data`) | Private, SSE-S3, HTTPS only. `backups/` objects expire after 30 days. Kept on `cdk destroy`. |
| S3 models bucket (`al-models`) | Same, plus versioning (old versions expire after 90 days). Kept on `cdk destroy`. |
| SQS job queue + dead-letter queue | 20 s long polling, 30 min visibility, 14-day retention, a job goes to the DLQ after 3 failed receives. |
| IAM user for the GPU worker | SSM port forwarding to the host only (no shell), consume jobs, report Step Functions task results, read `raw/` and `manifests/`, write `scores/`, read/write `mlflow-artifacts/`. No access key: you create it (section 4). |
| VPC + EC2 instance | One public subnet for the host, no NAT gateway. Amazon Linux 2023, `t3.medium` (2 vCPU, 4 GB), IMDSv2, no SSH key: connect with Session Manager. Docker, compose and cron installed, then Postgres, MLflow and Label Studio (behind Caddy and oauth2-proxy) started on first boot. |
| Elastic IP | The host's fixed public address, so Label Studio's URL and its Cognito callback survive stop and start. About $3.60 a month, charged also while the instance is stopped. |
| EBS data volume (`/data`) | gp3 20 GB, encrypted, mounted at `/data` on first boot. Kept when the instance or the stack is deleted. |
| Security group | 80 and 443 from anywhere (Label Studio through Caddy and the Cognito gate); Postgres (5432) and Label Studio's API (8080) from the Lambdas' security group only. You and the worker reach MLflow (5000) and Postgres (5432) through SSM port forwarding (section 3). |
| Lambda subnet | Private subnet on a second address range (10.1.0.0/24), no internet route; S3 through a free gateway endpoint. |
| `al-round` state machine + 7 Lambdas | One AL round: mark running, snapshot, train and score jobs on SQS, select, wait for labels (the round's Label Studio project, section 3b), merge; on any error the round is marked failed and its queued images go back to the pool. ARN in SSM `/leaf-al/state-machines/al-round`. Lambda logs kept 30 days. |
| SSM parameters | `/leaf-al/buckets/data`, `/leaf-al/buckets/models`, `/leaf-al/queues/jobs-url`, and for Label Studio `/leaf-al/auth/user-pool-id`, `/leaf-al/label-studio/domain`, `/leaf-al/label-studio/client-id`. |
| AWS Budget | Monthly cost budget, email at 80 % actual and 100 % forecast. Only created when `budgetEmail` is set. |
| Cognito user pool | Sign-in for people (section 6). Groups `admins`, `labelers`, `uploaders`; self sign-up off; email sign-in; optional TOTP MFA; hosted login domain; public PKCE clients `mobile` and `admin-web`, and the confidential client `label-studio` for the Label Studio gate. Kept on `cdk destroy`. |
| Post-confirmation Lambda | Adds users who signed in with Google to `uploaders`. Logs kept 30 days. |
| HTTP API + API Lambda | API Gateway HTTP API; every route needs a Cognito access token (JWT authorizer, 401 without one) except the app's public `GET /v1/crops` and `GET /v1/devices/config`; throttled to 10 requests/s (burst 20). One FastAPI Lambda (`api/`) checks groups; it runs in the Lambdas' subnet to use Postgres, signs download links for `edge/` in the models bucket, and signs upload links for `incoming/` in the data bucket (`POST /v1/captures`, `uploaders` only, at most `uploadDailyLimit` images per user in 24 hours, default 100). Labelers and admins read round progress (`GET /v1/rounds/active`, admins also `GET /v1/rounds/{id}`) and admins read dataset versions (`GET /v1/datasets/{name}/versions[/{n}]`); for a round's phase it reads `selections/` in the data bucket. While the EC2 host is stopped it answers `503`. Logs kept 30 days. |
| Label Studio webhook | `POST /v1/hooks/label-studio` on the same API, no token: Label Studio sends the shared secret header instead. A Lambda outside the VPC checks it, has a Lambda in the VPC store the label, and resumes the round once all k images are done. |
| Upload validation | App images land in `incoming/` of the data bucket; S3 notifies the `UploadQueue` (SQS), and `UploadFn` in the Lambdas' subnet checks each one (a whole JPEG or PNG of the declared type and SHA-256), turns it upright, strips EXIF, writes it to `raw/{dataset}/` and adds it to the train pool. Refused files are deleted. While the host is stopped, messages retry every 15 minutes for about a day, then wait 14 days in `UploadDlq`; unprocessed files in `incoming/` expire after 14 days. |

Deploy-time settings are CDK context values (`-c key=value`):

| Key | Default | Meaning |
| --- | --- | --- |
| `budgetEmail` | none (no Budget) | Where Budget alerts go |
| `budgetUsd` | `20` | Monthly budget in USD |
| `instanceType` | `t3.medium` | EC2 instance type (x86_64) |
| `dataVolumeGb` | `20` | Size of the `/data` volume |
| `authDomainPrefix` | `leaf-al-<account id>` | Prefix of the Cognito login domain (`https://<prefix>.auth.<region>.amazoncognito.com`) |
| `googleClientId` | none (no Google sign-in) | Google OAuth client id (section 6) |
| `mobileCallbackUrls` | `leafal://auth/callback` | Comma-separated sign-in redirects of the app |
| `adminCallbackUrls` | `http://localhost:8080/auth/callback` | Comma-separated sign-in redirects of admin tools |
| `labelStudioDomain` | `<Elastic IP>.sslip.io` | Label Studio's host name. The default needs no domain: sslip.io resolves it to the IP, and Caddy gets a Let's Encrypt certificate for it |

## 1. Secrets in SSM Parameter Store (before the first deploy)

CloudFormation cannot create SecureString parameters, so the Postgres password
is yours to create. Never commit it. Use URL-safe characters (it goes into the
MLflow backend URI):

```bash
aws ssm put-parameter --name /leaf-al/postgres/password --type SecureString \
  --value "$(openssl rand -hex 24)"
```

Postgres only accepts TLS connections from the network. Make its certificate
on your machine and store it next to the password: a private CA that clients
pin (`tls-ca`), and the server certificate it signed with its key (`tls-cert`,
`tls-key`). The key is a SecureString; the certificates are not secret:

```bash
deploy/ec2/make-tls.sh /tmp/leaf-al-tls
aws ssm put-parameter --overwrite --name /leaf-al/postgres/tls-key --type SecureString --value file:///tmp/leaf-al-tls/server.key
aws ssm put-parameter --overwrite --name /leaf-al/postgres/tls-cert --type String --value file:///tmp/leaf-al-tls/server.crt
aws ssm put-parameter --overwrite --name /leaf-al/postgres/tls-ca --type String --value file:///tmp/leaf-al-tls/ca.crt
rm -r /tmp/leaf-al-tls
```

To replace the certificate later (or the old self-signed one from before the
CA), run the same block, then `scripts/ec2.sh setup`: it restarts Postgres
when the certificate changed. Clients then download `tls-ca` again.

The round Lambdas have no internet access, so they cannot read SSM at run
time. CloudFormation copies the password into their settings at deploy time
from a Secrets Manager copy (about $0.40 a month). Create it once, before the
first deploy that includes the `al-round` state machine:

```bash
aws secretsmanager create-secret --name leaf-al/postgres-password --secret-string "$(aws ssm get-parameter --name /leaf-al/postgres/password --with-decryption --query Parameter.Value --output text)"
```

If you ever change the password, update both copies and deploy again.

Label Studio needs five values in one Secrets Manager secret (about $0.40 a
month): its admin account, the admin's API token (the label step Lambdas use
it), the webhook secret Label Studio sends back to the API, and the cookie
secret of the Cognito gate. Use your own email for `admin_email`. Create it
once, before the first deploy that includes Label Studio:

```bash
aws secretsmanager create-secret --name leaf-al/label-studio --secret-string "{\"admin_email\":\"you@example.com\",\"admin_password\":\"$(openssl rand -hex 16)\",\"api_token\":\"$(openssl rand -hex 20)\",\"webhook_secret\":\"$(openssl rand -hex 24)\",\"cookie_secret\":\"$(openssl rand -hex 16)\"}"
```

Read the admin password back when you need it:
`aws secretsmanager get-secret-value --secret-id leaf-al/label-studio --query SecretString --output text`.
Label Studio takes the account and token only on its first start; to change
them later, change them in Label Studio too.

The stack writes the bucket names and queue URL to SSM itself.

## 2. CDK stack

```bash
uv sync --extra infra
source .venv/bin/activate
npx aws-cdk bootstrap "aws://$CDK_DEFAULT_ACCOUNT/$AWS_REGION"   # once per account/region

CTX="-c budgetEmail=<your-email> -c budgetUsd=20"
npx aws-cdk diff   LeafAlPlatform $CTX                           # review what will change
npx aws-cdk deploy LeafAlPlatform $CTX
```

Use the same `-c` values on every deploy. A deploy without `budgetEmail`
deletes the Budget. The stack ignores the old `adminCidr` and `workerCidr`
settings: no IP address gets inbound access.

If you already created a budget named `leaf-al-monthly` by hand (the earlier
version of this doc did), delete it first, or the deploy fails on the name:
`aws budgets delete-budget --account-id "$CDK_DEFAULT_ACCOUNT" --budget-name leaf-al-monthly`.

AWS sends a confirmation email for the Budget alert subscription; confirm it.

The deploy prints these outputs; you need them below:
`DataBucketName`, `ModelsBucketName`, `JobQueueUrl`, `JobDlqUrl`,
`WorkerUserName`, `InstanceId`, `DataVolumeId`, `UserPoolId`,
`MobileClientId`, `AdminWebClientId`, `AuthDomainUrl`, `ApiUrl`,
`LabelStudioUrl`.

`cdk deploy` builds the API Lambda bundle with `uv` (Linux wheels, no Docker
needed); without `uv` on the PATH it falls back to Docker.

```bash
aws cloudformation describe-stacks --stack-name LeafAlPlatform \
  --query 'Stacks[0].Outputs' --output table
```

`cdk deploy` packages the Lambda code on your machine with `uv` (Linux
wheels, no Docker needed); it takes a minute the first time.

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

## 3. EC2 services (Postgres, MLflow, Label Studio)

Nothing to do by hand on a new instance. First boot mounts `/data`, clones this
repo to `/home/ec2-user/duytq5-leaf-al-platform` and runs
`deploy/ec2/bootstrap.sh`, which writes `.env` from SSM, runs
`docker compose up -d --build` and installs the nightly backup cron line. That
needs the Postgres password and the TLS parameters from section 1 to exist
before the deploy.

Check it a few minutes after the deploy (the first image build takes a while),
from your machine:

```bash
scripts/ec2.sh tunnel    # then open http://localhost:5000
scripts/ec2.sh backup    # in another terminal: "backed up to s3://..."
```

If MLflow does not open, look at the first-boot log, fix the cause (usually a
missing SSM parameter), and run the setup again:

```bash
scripts/ec2.sh shell     # needs the Session Manager plugin, see below
sudo tail -n 40 /var/log/cloud-init-output.log
```

```bash
scripts/ec2.sh setup     # from your machine: pull main on the host, rerun bootstrap.sh
```

Run `scripts/ec2.sh setup` again whenever `deploy/ec2/` changes on main, and
once on an instance created before first boot did this (it clones the repo if
it is missing). It is safe to repeat. The host's checkout only tracks main:
`setup` resets it to `origin/main`, so edits made on the host are dropped
(`.env` is not tracked and stays).

Changing `instanceType` on a deployed stack stops and starts the same instance
(`/data` and the root disk stay), but first boot does not run again, so run
`scripts/ec2.sh setup` after that deploy.

Session Manager needs its plugin for the AWS CLI on your machine, once:
`brew install --cask session-manager-plugin` on macOS, or see
[the install guide](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).
Without it, `shell` and `tunnel` fail with `SessionManagerPlugin is not found`.

### Reach MLflow and Postgres (SSM tunnel)

The host has no inbound ports. Open a tunnel with Session Manager port
forwarding instead. It works from any network, and CloudTrail logs each session.
Keep the command running while you use the tunnel:

```bash
aws ssm start-session --target <InstanceId> \
  --document-name AWS-StartPortForwardingSession \
  --parameters portNumber=5000,localPortNumber=5000
```

MLflow is then at `http://localhost:5000`. For Postgres, run a second session
with `portNumber=5432,localPortNumber=5432` and connect to `localhost:5432`.
The instance ID stays the same when the instance stops and starts.

### Daily start and stop

EC2 is the only part of the stack that costs money while idle, so stop the
instance when you are not working. The disks keep all data and Postgres and
MLflow start again on boot (`restart: unless-stopped`). Use the script, with
`AWS_PROFILE` and `AWS_REGION` set as above:

```bash
scripts/ec2.sh start             # waits until running
scripts/ec2.sh stop              # backs up Postgres to S3 first, then stops
scripts/ec2.sh stop --no-backup  # skip the backup
scripts/ec2.sh status            # instance state
scripts/ec2.sh backup            # backup only
scripts/ec2.sh setup             # pull main on the host, rerun bootstrap.sh
scripts/ec2.sh tunnel            # MLflow on localhost:5000 (Ctrl-C to close)
scripts/ec2.sh tunnel postgres   # Postgres on localhost:5432
scripts/ec2.sh tunnel mlflow 5050 # another local port, see below
scripts/ec2.sh shell             # Session Manager shell
```

`write-env.sh` refuses a Postgres password shorter than 32 characters.
Postgres rejects every network connection without TLS (`postgres/pg_hba.conf`),
MLflow's included, and the tunnel does not change that.

On macOS, port 5000 is taken by AirPlay Receiver (System Settings → General →
AirDrop & Handoff). Turn it off, or use `scripts/ec2.sh tunnel mlflow 5050` and
open `http://localhost:5050`.

`stop` runs the backup because the nightly cron backup only runs while the
instance is on. If the backup fails, the instance is left running. The backup
runs through SSM Run Command, so the repo must be cloned at
`/home/ec2-user/duytq5-leaf-al-platform` with `.env` written (section 3 does both on first boot).

While it is stopped, nothing that needs Postgres or MLflow can run; jobs sent
to SQS wait in the queue. Do not `cdk destroy` to save money day to day: the
next deploy creates new, empty buckets and a new data volume.

The tunnel is for you and the worker. The PR that adds the first Lambda that
needs Postgres also decides how that Lambda reaches it.

If the services already run from an earlier version of this repo, create the
three TLS parameters (section 1), then from your machine:

```bash
scripts/ec2.sh setup     # recreates Postgres with TLS; data is kept
```

## 3a. Database tables and seed data

Run from your own machine. First open the Postgres tunnel from section 3 in a
separate terminal and keep it running:

```bash
aws ssm start-session --target <InstanceId> \
  --document-name AWS-StartPortForwardingSession \
  --parameters portNumber=5432,localPortNumber=5432
```

Then get the CA certificate and build the connection string. The server
certificate names `localhost`, so `verify-full` works through the tunnel:

```bash
aws ssm get-parameter --name /leaf-al/postgres/tls-ca \
  --query Parameter.Value --output text > postgres-ca.crt
PG_PASSWORD=$(aws ssm get-parameter --name /leaf-al/postgres/password \
  --with-decryption --query Parameter.Value --output text)
export DATABASE_URL="postgresql://al:$PG_PASSWORD@localhost:5432/al?sslmode=verify-full&sslrootcert=$PWD/postgres-ca.crt"
export DATA_BUCKET=$(aws ssm get-parameter --name /leaf-al/buckets/data \
  --query Parameter.Value --output text)
```

Create the tables, then seed each dataset. `<image-dir>` has one sub-folder of
JPEGs per class, named by the class code in the dataset config (for RoCoLe:
`coffee___healthy`, `coffee___red_spider_mite`, `coffee___rust`). The dataset
config names its crop, which must be listed in `configs/crops.yaml`; the seed
adds or renames those crops. Check the split with `--dry-run` first; both
commands are safe to re-run if they stop half-way.

```bash
uv run al db migrate
uv run al seed configs/datasets/rocole.yaml <rocole-dir> --dry-run
uv run al seed configs/datasets/rocole.yaml <rocole-dir>
```

The seed uploads with your own AWS credentials: it needs `s3:PutObject` and
`s3:GetObject` on `raw/*` and `s3:ListBucket` on the data bucket (an admin
profile has them). There is no durian config yet: add
`configs/datasets/durian.yaml` once its class list is settled.

## 3b. Label Studio

Label Studio runs on the host next to Postgres and MLflow, with its projects
in a `labelstudio` database (so the nightly backup covers it). The way in:

```
browser -> https://<LabelStudioUrl> (Caddy, Let's Encrypt)
        -> oauth2-proxy: Cognito sign-in, only groups labelers and admins
        -> Label Studio sign-in with the person's own Label Studio account
```

Two sign-ins, on purpose (design doc decision 2026-10-05): Label Studio
Community has no SSO. `scripts/ec2.sh setup` (section 3) starts it all after
the deploy; Caddy needs a minute for the certificate the first time.

1. Open `LabelStudioUrl`, sign in with your Cognito admin account (section 6),
   then with `admin_email` and `admin_password` from the secret.
2. For each labeler: create the Cognito account in group `labelers`
   (section 6), then in Label Studio open Organization, "Add People", and send
   them the invite link. Open sign-up is off; the link is the only way in.

You never create projects by hand. When a round reaches its label step, the
step makes a project named `ACTIVE <dataset> <round id>` with one task per
selected image and the round model's guess preselected. Labelers confirm or
correct each one, or pick "Không phải lá" (not a leaf). Every submit is copied
to the `labels` table; when all k images are done the round goes on to merge
and the project is renamed `DONE ...`. Edits after submit are not recorded,
and skipping is turned off, so each task needs an answer.

If the label step fails with `LabelStudioError`, the instance was probably
stopped or Label Studio was restarting; the step retries for a few minutes on
its own. Check the containers with `scripts/ec2.sh shell`, then
`cd ~/duytq5-leaf-al-platform/deploy/ec2 && docker compose ps`.

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

The worker reaches MLflow through the same tunnel. Install the Session Manager
plugin on the GPU machine and run the port-forwarding command from section 3
with `--profile leaf-al-worker`. Then set `MLFLOW_TRACKING_URI=http://localhost:5000`.
The worker user can open port-forwarding sessions to this one instance only. It
cannot open a shell.

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

Postgres is dumped to S3 every night at 03:00 UTC by the cron line that
`bootstrap.sh` installs for `ec2-user` (log: `/home/ec2-user/leaf-al-backup.log`).

Dumps are kept for 30 days. Restore one into a fresh, empty Postgres:

```bash
aws s3 cp s3://<DataBucketName>/backups/postgres/<file>.sql.gz - | gunzip \
  | docker compose exec -T postgres psql -U al -d postgres
```

## 6. Sign-in (Cognito)

Cognito signs people in; machines keep IAM. Who belongs where:

| Group | Who | How they get in |
| --- | --- | --- |
| `admins` | You | You create the account (below) |
| `labelers` | People labeling in Label Studio | You create the account |
| `uploaders` | App users who send images | Automatic on their first Google sign-in |

Operators and ML engineers are GitHub roles (CODEOWNERS on the repo), not
Cognito groups.

### Create an admin or labeler

```bash
POOL_ID=$(aws cloudformation describe-stacks --stack-name LeafAlPlatform --query "Stacks[0].Outputs[?OutputKey=='UserPoolId'].OutputValue" --output text)
aws cognito-idp admin-create-user --user-pool-id "$POOL_ID" --username you@example.com --user-attributes Name=email,Value=you@example.com Name=email_verified,Value=true
aws cognito-idp admin-add-user-to-group --user-pool-id "$POOL_ID" --username you@example.com --group-name admins
```

Cognito emails a temporary password; the first sign-in asks for a new one.
Turn on TOTP MFA for admin accounts from the login page after first sign-in.
Use `--group-name labelers` for labelers.

### Google sign-in for the app (when the app is ready)

1. In Google Cloud Console, create an OAuth client of type "Web application".
   Authorized redirect URI: `<AuthDomainUrl>/oauth2/idpresponse`.
2. Store its client secret in Secrets Manager (about $0.40 a month):

   ```bash
   aws secretsmanager create-secret --name leaf-al/google-client-secret --secret-string '<client secret>'
   ```

3. Deploy with `-c googleClientId=<client id>` added to `CTX`, and keep it on
   every later deploy (without it, Google sign-in is removed).

### Check the API

```bash
API_URL=$(aws cloudformation describe-stacks --stack-name LeafAlPlatform --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)
curl -i "$API_URL/v1/me"
```

Without a token the answer is `401` from API Gateway. With an access token
from the login page (`Authorization: Bearer <token>`), `/v1/me` returns your
`sub`, username and groups.

The app's routes need no token and read Postgres, so start the host first:

```bash
curl -s "$API_URL/v1/crops"
curl -s "$API_URL/v1/devices/config?dataset=rocole"
```

`/v1/crops` lists only datasets with a released model, so it is `{"crops": []}`
and `/v1/devices/config` is `404` until the first model release.

The admin routes need an access token of a user in `admins` (or `labelers` for
the active rounds):

```bash
curl -s -H "Authorization: Bearer $TOKEN" "$API_URL/v1/rounds/active"
curl -s -H "Authorization: Bearer $TOKEN" "$API_URL/v1/datasets/rocole/versions"
```

### Uploads stuck in the DLQ

If the host was stopped for more than a day, some uploads wait in the DLQ.
With the host running, move them back:

```bash
DLQ_URL=$(aws cloudformation describe-stacks --stack-name LeafAlPlatform --query "Stacks[0].Outputs[?OutputKey=='UploadDlqUrl'].OutputValue" --output text)
aws sqs start-message-move-task --source-arn $(aws sqs get-queue-attributes --queue-url "$DLQ_URL" --attribute-names QueueArn --query Attributes.QueueArn --output text)
```

## 7. Later

The worker's train and score jobs and the GitHub start-round workflow are
added by later PRs; this file gets their steps as they land.
