"""scripts/ec2.sh, run against a stub `aws` on PATH (no AWS needed)."""

import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ec2.sh"

# Answers just enough of the AWS CLI for the script; logs every call.
STUB = """#!/usr/bin/env bash
echo "$*" >> "$AWS_LOG"
case "$1 $2" in
  "cloudformation describe-stacks") echo i-0abc ;;
  "ec2 describe-instances")
    echo "$STUB_STATE" ;;
  "ssm send-command") echo cmd-1 ;;
  "ssm get-command-invocation")
    case "$*" in
      *"--query Status"*)
        # STUB_BACKUP is a space-separated list of statuses, one per call.
        n=$(grep -c -- "--query Status" "$AWS_LOG")
        set -- $STUB_BACKUP
        shift $(( n > $# ? $# - 1 : n - 1 ))
        echo "$1" ;;
      *) echo "backed up" ;;
    esac ;;
esac
"""


def run(tmp_path, *args, state="running", backup="Success"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    aws = bin_dir / "aws"
    aws.write_text(STUB)
    aws.chmod(aws.stat().st_mode | stat.S_IEXEC)
    sleep = bin_dir / "sleep"  # the poll loop sleeps between checks
    sleep.write_text("#!/bin/sh\n")
    sleep.chmod(sleep.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "aws.log"
    log.write_text("")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "AWS_LOG": str(log),
        "STUB_STATE": state,
        "STUB_BACKUP": backup,
        "AWS_REGION": "ap-southeast-2",
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, check=False
    )
    return proc, log.read_text().splitlines()


def test_unknown_command_prints_usage_without_calling_aws(tmp_path):
    proc, calls = run(tmp_path, "bogus")
    assert proc.returncode == 1
    assert "scripts/ec2.sh start" in proc.stderr
    assert calls == []


def test_start_waits_until_running(tmp_path):
    proc, calls = run(tmp_path, "start")
    assert proc.returncode == 0, proc.stderr
    assert any(c.startswith("ec2 start-instances --instance-ids i-0abc") for c in calls)
    assert any(c.startswith("ec2 wait instance-running") for c in calls)
    assert "tunnel" in proc.stdout


def test_tunnel_forwards_mlflow_by_default(tmp_path):
    proc, calls = run(tmp_path, "tunnel")
    assert proc.returncode == 0, proc.stderr
    assert calls[-1] == (
        "ssm start-session --target i-0abc --document-name AWS-StartPortForwardingSession"
        " --parameters portNumber=5000,localPortNumber=5000"
    )


def test_tunnel_custom_local_port(tmp_path):
    proc, calls = run(tmp_path, "tunnel", "mlflow", "5050")
    assert proc.returncode == 0, proc.stderr
    assert calls[-1].endswith("portNumber=5000,localPortNumber=5050")


def test_tunnel_postgres(tmp_path):
    proc, calls = run(tmp_path, "tunnel", "postgres")
    assert proc.returncode == 0, proc.stderr
    assert calls[-1].endswith("portNumber=5432,localPortNumber=5432")


def test_stop_backs_up_before_stopping(tmp_path):
    proc, calls = run(tmp_path, "stop")
    assert proc.returncode == 0, proc.stderr
    send = next(i for i, c in enumerate(calls) if c.startswith("ssm send-command"))
    stop = next(i for i, c in enumerate(calls) if c.startswith("ec2 stop-instances"))
    assert send < stop
    assert "backup-postgres.sh" in calls[send]


def test_setup_clones_or_pulls_then_runs_bootstrap(tmp_path):
    proc, calls = run(tmp_path, "setup")
    assert proc.returncode == 0, proc.stderr
    (send,) = [c for c in calls if c.startswith("ssm send-command")]
    assert "git clone https://github.com/duytq5/duytq5-leaf-al-platform.git" in send
    assert "pull --ff-only" in send
    assert send.index("mkswap /swapfile") < send.index("deploy/ec2/bootstrap.sh")
    assert "env AWS_REGION=ap-southeast-2 " in send
    assert send.index("git clone") < send.index("deploy/ec2/bootstrap.sh")


def test_backup_waits_while_command_runs(tmp_path):
    proc, calls = run(tmp_path, "backup", backup="InProgress Success")
    assert proc.returncode == 0, proc.stderr
    status_calls = [c for c in calls if "--query Status" in c]
    assert len(status_calls) == 2


def test_stop_aborts_when_backup_fails(tmp_path):
    proc, calls = run(tmp_path, "stop", backup="Failed")
    assert proc.returncode == 1
    assert "--no-backup" in proc.stderr
    assert not any(c.startswith("ec2 stop-instances") for c in calls)


def test_stop_no_backup_skips_backup(tmp_path):
    proc, calls = run(tmp_path, "stop", "--no-backup")
    assert proc.returncode == 0, proc.stderr
    assert not any(c.startswith("ssm send-command") for c in calls)
    assert any(c.startswith("ec2 stop-instances") for c in calls)


def test_stop_skips_backup_when_already_stopped(tmp_path):
    proc, calls = run(tmp_path, "stop", state="stopped")
    assert proc.returncode == 0, proc.stderr
    assert not any(c.startswith("ssm send-command") for c in calls)
