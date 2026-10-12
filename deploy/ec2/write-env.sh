#!/bin/sh
# Run on the EC2 instance: writes deploy/ec2/.env and the Postgres TLS files
# in deploy/ec2/tls/ from SSM Parameter Store.
# The instance role needs ssm:GetParameter (+ kms:Decrypt for SecureString).
set -eu
PREFIX="${SSM_PREFIX:-/leaf-al}"
REGION="${AWS_REGION:?set AWS_REGION}"

get() {
  aws ssm get-parameter --region "$REGION" --with-decryption \
    --name "$PREFIX/$1" --query Parameter.Value --output text
}

umask 077
password=$(get postgres/password)
# Postgres is reachable from the internet (Lambdas run outside a VPC).
if [ "${#password}" -lt 32 ]; then
  echo "$PREFIX/postgres/password is shorter than 32 characters; use openssl rand -hex 24" >&2
  exit 1
fi

dir="$(dirname "$0")"
mkdir -p "$dir/tls"
get postgres/tls-key > "$dir/tls/server.key"
get postgres/tls-cert > "$dir/tls/server.crt"
chmod 644 "$dir/tls/server.crt"

cat > "$dir/.env" <<EOF
POSTGRES_USER=al
POSTGRES_PASSWORD=$password
POSTGRES_DB=al
MLFLOW_DB=mlflow
AWS_REGION=$REGION
MODELS_BUCKET=$(get buckets/models)
DATA_BUCKET=$(get buckets/data)
DATA_DIR=/data
EOF
echo "wrote $dir/.env and $dir/tls/"
