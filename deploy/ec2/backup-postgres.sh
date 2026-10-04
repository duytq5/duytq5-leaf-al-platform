#!/bin/sh
# Run on the EC2 instance (from cron): dump all Postgres databases to S3.
# Covers the metadata db and the MLflow db. Retry-safe: each run writes a new
# timestamped object. Needs s3:PutObject on the data bucket's backups/ prefix.
set -eu
cd "$(dirname "$0")"
. ./.env

key="backups/postgres/$(date -u +%Y%m%dT%H%M%SZ).sql.gz"
docker compose exec -T postgres pg_dumpall -U "$POSTGRES_USER" \
  | gzip \
  | aws s3 cp --region "$AWS_REGION" - "s3://$DATA_BUCKET/$key"
echo "backed up to s3://$DATA_BUCKET/$key"
