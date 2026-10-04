# leaf-al-platform

Proof-of-concept active-learning (AL) MLOps platform for leaf-disease image
classification (RoCoLe coffee and a Vietnamese durian dataset, FastViT backbone).
AWS orchestrates and stores, a local RTX 5070 trains and scores, and a mobile app
uploads images and downloads ONNX models.

## Layout

| Path | What lives there |
| --- | --- |
| `common/` | Shared Pydantic models: SQS job messages, Edge API bodies, model manifest, training config, S3 key layout |
| `selection/` | Selection strategy registry (`@register("name")`) and the built-in strategies |
| `worker/` | Local GPU worker: long-polls SQS, runs `train` / `score` / `export` jobs |
| `lambdas/` | Lambda handlers (select, oracle; later edge API, validation, Label Studio webhook) |
| `infra/` | AWS CDK app (Python), one stack: `LeafAlPlatform` |
| `deploy/ec2/` | Docker Compose for the single EC2 instance: Postgres + MLflow |
| `configs/train/` | YAML training configs |
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

### CDK

```bash
source .venv/bin/activate
npx aws-cdk synth            # nothing is deployed by synth
```

### EC2 services

```bash
cd deploy/ec2
cp .env.example .env         # fill in; on EC2 generate it from SSM Parameter Store
docker compose up -d
```

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
