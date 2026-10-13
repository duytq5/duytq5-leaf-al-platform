#!/bin/sh
# Run on the EC2 instance as ec2-user: bring up Postgres, MLflow, Label Studio
# (behind Caddy and oauth2-proxy) and the nightly backup. First boot runs it
# from user data; `scripts/ec2.sh setup` runs it on an existing instance.
# Safe to run again: .env is rewritten from SSM, compose only recreates what
# changed, Postgres restarts only for a new certificate, the Label Studio
# database is created once, the cron line is replaced.
set -eu
REGION="${AWS_REGION:?set AWS_REGION}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LOG="$(dirname "$REPO")/leaf-al-backup.log"

cd "$HERE"
AWS_REGION="$REGION" ./write-env.sh
docker compose up -d --wait postgres
# Postgres reads its certificate only when it starts (postgres/start.sh), so
# restart it once after write-env.sh brought a new one from SSM.
cert="$(sha256sum tls/server.crt | cut -d' ' -f1)"
if [ "$(cat tls/.running 2>/dev/null || true)" != "$cert" ]; then
  docker compose restart postgres
  docker compose up -d --wait postgres
  echo "$cert" > tls/.running
fi
# Label Studio keeps its projects in Postgres too (nightly backup included);
# its database must exist before it starts. CREATE DATABASE has no IF NOT EXISTS.
psql_al() { docker compose exec -T postgres psql -U al -d al -v ON_ERROR_STOP=1 -tAc "$1"; }
if [ "$(psql_al "SELECT 1 FROM pg_database WHERE datname = 'labelstudio'")" != "1" ]; then
  psql_al "CREATE DATABASE labelstudio"
fi
docker compose up -d --build

line="0 3 * * * $HERE/backup-postgres.sh >> $LOG 2>&1"
others="$(crontab -l 2>/dev/null | grep -vF backup-postgres.sh || true)"
{ [ -z "$others" ] || echo "$others"; echo "$line"; } | crontab -
echo "services up, nightly backup at 03:00 UTC"
