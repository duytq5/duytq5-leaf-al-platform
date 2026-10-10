#!/usr/bin/env bash
# Start and stop the platform's EC2 host (Postgres + MLflow) to save cost.
# Run from your own machine with your AWS profile; see docs/aws-setup.md.
#
#   scripts/ec2.sh start          start the instance, wait, print the MLflow URL
#   scripts/ec2.sh stop           back up Postgres to S3, then stop the instance
#   scripts/ec2.sh stop --no-backup
#   scripts/ec2.sh backup         back up Postgres to S3 (instance must be running)
#   scripts/ec2.sh status         instance state and public IP
#   scripts/ec2.sh shell          open a Session Manager shell on the instance
#
# Uses AWS_PROFILE and AWS_REGION from the environment. Set INSTANCE_ID to skip
# the CloudFormation lookup.
set -euo pipefail

STACK=LeafAlPlatform
REPO_DIR=/home/ec2-user/duytq5-leaf-al-platform
MLFLOW_PORT=5000

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

instance_id() {
  if [ -n "${INSTANCE_ID:-}" ]; then
    echo "$INSTANCE_ID"
    return
  fi
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text
}

state() {
  aws ec2 describe-instances --instance-ids "$1" \
    --query 'Reservations[0].Instances[0].State.Name' --output text
}

public_ip() {
  aws ec2 describe-instances --instance-ids "$1" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

backup() {
  local id=$1 cmd status
  echo "Backing up Postgres to S3..."
  cmd=$(aws ssm send-command --instance-ids "$id" \
    --document-name AWS-RunShellScript \
    --comment "leaf-al postgres backup" \
    --parameters "commands=[\"sudo -u ec2-user $REPO_DIR/deploy/ec2/backup-postgres.sh\"]" \
    --query 'Command.CommandId' --output text)
  # The waiter exits non-zero on failure; read the result either way.
  aws ssm wait command-executed --command-id "$cmd" --instance-id "$id" || true
  status=$(aws ssm get-command-invocation --command-id "$cmd" --instance-id "$id" \
    --query 'Status' --output text)
  aws ssm get-command-invocation --command-id "$cmd" --instance-id "$id" \
    --query '[StandardOutputContent, StandardErrorContent]' --output text
  if [ "$status" != "Success" ]; then
    echo "Backup failed ($status)." >&2
    return 1
  fi
}

cmd=${1:-}
case "$cmd" in
  -h | --help | help) usage 0 ;;
  start | stop | backup | status | shell) ;;
  *) usage 1 >&2 ;;
esac

id=$(instance_id)
if [ -z "$id" ] || [ "$id" = "None" ]; then
  echo "No InstanceId output on stack $STACK. Is it deployed in ${AWS_REGION:-this region}?" >&2
  exit 1
fi

case "$cmd" in
  start)
    aws ec2 start-instances --instance-ids "$id" >/dev/null
    echo "Starting $id..."
    aws ec2 wait instance-running --instance-ids "$id"
    ip=$(public_ip "$id")
    echo "Running. Public IP: $ip (changes on every start)"
    echo "MLflow: http://$ip:$MLFLOW_PORT (Postgres and MLflow take a minute to come up)"
    ;;
  stop)
    if [ "${2:-}" != "--no-backup" ] && [ "$(state "$id")" = "running" ]; then
      if ! backup "$id"; then
        echo "Not stopping. Fix the backup, or run: $0 stop --no-backup" >&2
        exit 1
      fi
    fi
    aws ec2 stop-instances --instance-ids "$id" >/dev/null
    echo "Stopping $id..."
    aws ec2 wait instance-stopped --instance-ids "$id"
    echo "Stopped. Data on /data is kept; only the disks are billed now."
    ;;
  backup)
    backup "$id"
    ;;
  status)
    echo "$id: $(state "$id"), public IP $(public_ip "$id")"
    ;;
  shell)
    exec aws ssm start-session --target "$id"
    ;;
esac
