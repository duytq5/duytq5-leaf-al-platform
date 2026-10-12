#!/usr/bin/env bash
# Start and stop the platform's EC2 host (Postgres + MLflow) to save cost.
# Run from your own machine with your AWS profile; see docs/aws-setup.md.
#
#   scripts/ec2.sh start          start the instance and wait until it runs
#   scripts/ec2.sh stop           back up Postgres to S3, then stop the instance
#   scripts/ec2.sh stop --no-backup
#   scripts/ec2.sh backup         back up Postgres to S3 (instance must be running)
#   scripts/ec2.sh setup          clone or pull main on the host, then run
#                                 deploy/ec2/bootstrap.sh (Postgres, MLflow, cron)
#   scripts/ec2.sh status         instance state
#   scripts/ec2.sh tunnel         MLflow on localhost:5000 via SSM (Ctrl-C closes)
#   scripts/ec2.sh tunnel postgres
#                                 Postgres on localhost:5432 via SSM
#   scripts/ec2.sh tunnel mlflow 5050
#                                 another local port (macOS AirPlay uses 5000)
#   scripts/ec2.sh shell          open a Session Manager shell on the instance
#
# Uses AWS_PROFILE and AWS_REGION from the environment. Set INSTANCE_ID to skip
# the CloudFormation lookup.
set -euo pipefail

STACK=LeafAlPlatform
REPO_DIR=/home/ec2-user/duytq5-leaf-al-platform
REPO_URL=https://github.com/duytq5/duytq5-leaf-al-platform.git
MLFLOW_PORT=5000
POSTGRES_PORT=5432

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
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

# Run shell commands on the host as root through SSM Run Command and print
# their output. Arguments: a label, then one or more commands (no double quotes).
run_on_host() {
  local id=$1 label=$2 cmd status params
  shift 2
  params=$(printf '"%s",' "$@")
  echo "$label..."
  cmd=$(aws ssm send-command --instance-ids "$id" \
    --document-name AWS-RunShellScript \
    --comment "leaf-al $label" \
    --parameters "commands=[${params%,}]" \
    --query 'Command.CommandId' --output text)
  # Poll instead of `aws ssm wait`, which gives up after 100 s; a first
  # `docker compose up --build` takes longer.
  while :; do
    status=$(aws ssm get-command-invocation --command-id "$cmd" --instance-id "$id" \
      --query 'Status' --output text 2>/dev/null || echo Pending)
    case "$status" in Pending | InProgress | Delayed) sleep 5 ;; *) break ;; esac
  done
  aws ssm get-command-invocation --command-id "$cmd" --instance-id "$id" \
    --query '[StandardOutputContent, StandardErrorContent]' --output text
  if [ "$status" != "Success" ]; then
    echo "$label failed ($status)." >&2
    return 1
  fi
}

backup() {
  run_on_host "$1" "Backing up Postgres to S3" \
    "sudo -u ec2-user $REPO_DIR/deploy/ec2/backup-postgres.sh"
}

setup() {
  run_on_host "$1" "Setting up Postgres, MLflow and the nightly backup" \
    "if [ -d $REPO_DIR/.git ]; then sudo -u ec2-user git -C $REPO_DIR pull --ff-only; else sudo -u ec2-user git clone $REPO_URL $REPO_DIR; fi" \
    "sudo -u ec2-user env AWS_REGION=${AWS_REGION:?set AWS_REGION} $REPO_DIR/deploy/ec2/bootstrap.sh"
}

cmd=${1:-}
case "$cmd" in
  -h | --help | help) usage 0 ;;
  start | stop | backup | setup | status | tunnel | shell) ;;
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
    echo "Running. Postgres and MLflow take a minute to come up."
    echo "Open MLflow with: $0 tunnel"
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
  setup)
    setup "$id"
    ;;
  status)
    echo "$id: $(state "$id")"
    ;;
  tunnel)
    case "${2:-mlflow}" in
      mlflow) port=$MLFLOW_PORT ;;
      postgres) port=$POSTGRES_PORT ;;
      *) usage 1 >&2 ;;
    esac
    local_port=${3:-$port}
    echo "Forwarding localhost:$local_port to port $port on the instance. Ctrl-C to close."
    exec aws ssm start-session --target "$id" \
      --document-name AWS-StartPortForwardingSession \
      --parameters "portNumber=$port,localPortNumber=$local_port"
    ;;
  shell)
    exec aws ssm start-session --target "$id"
    ;;
esac
