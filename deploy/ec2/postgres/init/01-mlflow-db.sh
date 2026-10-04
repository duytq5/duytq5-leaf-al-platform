#!/bin/sh
# Runs once, on first start of an empty volume: a separate database for MLflow.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
CREATE DATABASE "${MLFLOW_DB}";
EOSQL
