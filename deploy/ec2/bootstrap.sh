#!/bin/sh
# Run on the EC2 instance as ec2-user: bring up Postgres + MLflow and the
# nightly backup. First boot runs it from user data; `scripts/ec2.sh setup`
# runs it on an existing instance. Safe to run again: .env is rewritten from
# SSM, compose only recreates what changed, the cron line is replaced.
set -eu
REGION="${AWS_REGION:?set AWS_REGION}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LOG="$(dirname "$REPO")/leaf-al-backup.log"

cd "$HERE"
AWS_REGION="$REGION" ./write-env.sh
docker compose up -d --build

line="0 3 * * * $HERE/backup-postgres.sh >> $LOG 2>&1"
others="$(crontab -l 2>/dev/null | grep -vF backup-postgres.sh || true)"
{ [ -z "$others" ] || echo "$others"; echo "$line"; } | crontab -
echo "services up, nightly backup at 03:00 UTC"
