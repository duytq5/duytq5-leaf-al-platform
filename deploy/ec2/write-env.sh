#!/bin/sh
# Run on the EC2 instance: writes deploy/ec2/.env from SSM Parameter Store.
# The instance role needs ssm:GetParameter (+ kms:Decrypt for SecureString).
set -eu
PREFIX="${SSM_PREFIX:-/leaf-al}"
REGION="${AWS_REGION:?set AWS_REGION}"

get() {
  aws ssm get-parameter --region "$REGION" --with-decryption \
    --name "$PREFIX/$1" --query Parameter.Value --output text
}

umask 077
cat > "$(dirname "$0")/.env" <<EOF
POSTGRES_USER=al
POSTGRES_PASSWORD=$(get postgres/password)
POSTGRES_DB=al
MLFLOW_DB=mlflow
AWS_REGION=$REGION
MODELS_BUCKET=$(get buckets/models)
DATA_BUCKET=$(get buckets/data)
DATA_DIR=/data
EOF
echo "wrote $(dirname "$0")/.env"
