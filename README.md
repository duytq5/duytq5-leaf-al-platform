# leaf-al-platform

Proof-of-concept active-learning (AL) MLOps platform for leaf-disease image
classification (RoCoLe coffee and a Vietnamese durian dataset, FastViT backbone).
AWS orchestrates and stores, a local RTX 5070 trains and scores, and a mobile app
uploads images and downloads ONNX models.

## Layout

| Path | What lives there |
| --- | --- |
| `common/` | Shared Pydantic models: SQS job messages, Edge API bodies, model manifest, training config, S3 key layout |
| `db/` | Postgres schema (`db/migrations/*.sql`), migration runner, dataset seed script |
| `selection/` | Selection strategy registry (`@register("name")`) and the built-in strategies |
| `worker/` | Local GPU worker: long-polls SQS, runs `train` / `score` / `export` jobs |
| `lambdas/` | Lambda handlers (select, oracle; later edge API, validation, Label Studio webhook) |
| `infra/` | AWS CDK app (Python), one stack: `LeafAlPlatform` |
| `deploy/ec2/` | Docker Compose for the single EC2 instance: Postgres + MLflow |
| `cli/` | Operator CLI `al` |
| `configs/datasets/` | YAML dataset configs: the label list and the fixed train/val/test split |
| `configs/train/` | YAML training configs |
| `configs/rounds/` | YAML round configs: strategy, its parameters and k, chosen by a human per round |
| `tests/` | Unit tests |

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra infra        # core + dev tools + CDK
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Extras: `worker` (torch, timm, mlflow, boto3), `lambdas` (boto3), `infra` (aws-cdk-lib).
On the RTX 5070 machine install torch from the CUDA 12.8 index:
`uv pip install torch --index-url https://download.pytorch.org/whl/cu128`.

No AWS credentials are needed to build, lint or test. CDK synth also runs
offline: `source .venv/bin/activate && npx aws-cdk synth`.

### Local services

```bash
cd deploy/ec2
cp .env.example .env         # local values only; never commit .env
./make-tls.sh                # self-signed Postgres certificate in ./tls (gitignored)
docker compose up -d
```

Postgres only accepts network connections over TLS, MLflow's included. To run
the database tests against it:

```bash
. deploy/ec2/.env
export TEST_DATABASE_URL="postgresql://al:$POSTGRES_PASSWORD@127.0.0.1:5432/al?sslmode=verify-ca&sslrootcert=deploy/ec2/tls/server.crt"
uv run pytest                # without TEST_DATABASE_URL the database tests are skipped
```

### Database and seed data

```bash
export DATABASE_URL=...      # as above, or the EC2 one from docs/aws-setup.md
uv run al db migrate         # create or update the tables; safe to re-run
uv run al seed configs/datasets/rocole.yaml <image-dir> --dry-run   # show the split
uv run al seed configs/datasets/rocole.yaml <image-dir> --bucket <data-bucket>
```

`<image-dir>` has one sub-folder of JPEGs per label, named as in the dataset
config. The seed uploads each image to `raw/<dataset>/<sha256>.jpg`, then records
it with a split that is stratified per class and fixed by the split seed.
Re-running it skips what is already there, and an image keeps its first split.
Train images become the AL pool (`unlabeled`), with their ground truth only in
`oracle_labels`. Val and test images are `labeled` and can never enter the pool
(a database constraint enforces it).

### AWS

Nothing is deployed from this repo automatically. The owner runs the AWS
steps by hand; the exact commands are in [docs/aws-setup.md](docs/aws-setup.md).

## Choosing a strategy per round

Selection is never hardcoded. Before each round you look at the data and the
latest scores, then write a round config:

```yaml
dataset: durian
mode: simulation
train_config: configs/train/example.yaml
selection: {strategy: random, params: {}, k: 50, seed: 42}
```

```bash
uv run al strategies                                   # names, required inputs, parameters
uv run al check-round configs/rounds/example.yaml      # validate before starting
```

Unknown strategies, misspelled parameters and invalid k are rejected. The
chosen strategy, parameters and k are stored on the round's row, so every
round can be reproduced. Phase 1 adds `al round preview` (dry-run a strategy
on the latest scores: which images it would pick, their predicted classes and
confidence) and `al round start`.

## Rules

- POC scope: no services or features beyond the design doc without the owner's say.
- Jobs and Lambdas are retry-safe (SQS can deliver twice; the app retries uploads).
- Test-split images never enter the AL pool or Label Studio.
- One label list per dataset; its order is the model's output order and is copied into the manifest.
- Buckets are private, uploads land in `incoming/` first, secrets live in SSM Parameter Store and are never committed.
- Cost: the EC2 instance is the only always-on resource; stop it when idle. Lambdas run outside a VPC (no NAT gateway).

## Roadmap

1. **AL loop in simulation mode**: a 10-round RoCoLe simulation runs unattended for each strategy and exports learning curves.
2. **Labeling and edge upload**: an app-uploaded image is labeled in Label Studio in the next round and used in training.
3. **Model update to mobile**: promote makes the app download the new model; rollback restores the previous one.
