#!/bin/sh
# Run on the EC2 instance: writes deploy/ec2/.env and the Postgres TLS files
# in deploy/ec2/tls/ from SSM Parameter Store, the Label Studio secret from
# Secrets Manager and the Cognito client secret of the Label Studio gate.
# The instance role needs ssm:GetParameter (+ kms:Decrypt for SecureString),
# secretsmanager:GetSecretValue and cognito-idp:DescribeUserPoolClient.
set -eu
PREFIX="${SSM_PREFIX:-/leaf-al}"
REGION="${AWS_REGION:?set AWS_REGION}"

get() {
  aws ssm get-parameter --region "$REGION" --with-decryption \
    --name "$PREFIX/$1" --query Parameter.Value --output text
}

umask 077
password=$(get postgres/password)
# A long password, even though only the Lambdas' security group can reach Postgres.
if [ "${#password}" -lt 32 ]; then
  echo "$PREFIX/postgres/password is shorter than 32 characters; use openssl rand -hex 24" >&2
  exit 1
fi

dir="$(dirname "$0")"
mkdir -p "$dir/tls"
get postgres/tls-key > "$dir/tls/server.key"
get postgres/tls-cert > "$dir/tls/server.crt"
chmod 644 "$dir/tls/server.crt"

# Label Studio and its Cognito gate (oauth2-proxy). The stack writes the
# domain and client id to SSM; the owner made the secret (docs/aws-setup.md).
ls_json=$(aws secretsmanager get-secret-value --region "$REGION" \
  --secret-id "${LABEL_STUDIO_SECRET:-leaf-al/label-studio}" --query SecretString --output text)
ls_field() {
  printf '%s' "$ls_json" | python3 -c '
import json, sys
value = json.load(sys.stdin).get(sys.argv[1])
if not value:
    sys.exit(f"leaf-al/label-studio has no {sys.argv[1]}; see docs/aws-setup.md")
print(value)' "$1"
}
ls_admin_email=$(ls_field admin_email)
ls_admin_password=$(ls_field admin_password)
ls_api_token=$(ls_field api_token)
ls_field webhook_secret > /dev/null  # the Lambdas use it; checked here so a gap shows early
ls_cookie_secret=$(ls_field cookie_secret)
pool_id=$(get auth/user-pool-id)
client_id=$(get label-studio/client-id)
client_secret=$(aws cognito-idp describe-user-pool-client --region "$REGION" \
  --user-pool-id "$pool_id" --client-id "$client_id" \
  --query UserPoolClient.ClientSecret --output text)

cat > "$dir/.env" <<EOF
COMPOSE_FILE=docker-compose.yml:docker-compose.logs.yml
POSTGRES_USER=al
POSTGRES_PASSWORD=$password
POSTGRES_DB=al
MLFLOW_DB=mlflow
AWS_REGION=$REGION
MODELS_BUCKET=$(get buckets/models)
DATA_BUCKET=$(get buckets/data)
DATA_DIR=/data
LS_DOMAIN=$(get label-studio/domain)
VIEWER_DOMAIN=$(get viewer/domain)
LS_ADMIN_EMAIL=$ls_admin_email
LS_ADMIN_PASSWORD=$ls_admin_password
LS_API_TOKEN=$ls_api_token
OAUTH2_PROXY_OIDC_ISSUER_URL=https://cognito-idp.$REGION.amazonaws.com/$pool_id
OAUTH2_PROXY_CLIENT_ID=$client_id
OAUTH2_PROXY_CLIENT_SECRET=$client_secret
OAUTH2_PROXY_COOKIE_SECRET=$ls_cookie_secret
EOF
echo "wrote $dir/.env and $dir/tls/"
