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
    case "$*" in *State.Name*) echo "$STUB_STATE" ;; *) echo 203.0.113.7 ;; esac ;;
  "ssm send-command") echo cmd-1 ;;
  "ssm get-command-invocation")
    case "$*" in *"--query Status"*) echo "$STUB_BACKUP" ;; *) echo "backed up" ;; esac ;;
esac
"""


def run(tmp_path, *args, state="running", backup="Success"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    aws = bin_dir / "aws"
    aws.write_text(STUB)
    aws.chmod(aws.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "aws.log"
    log.write_text("")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "AWS_LOG": str(log),
        "STUB_STATE": state,
        "STUB_BACKUP": backup,
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


def test_start_waits_and_prints_mlflow_url(tmp_path):
    proc, calls = run(tmp_path, "start")
    assert proc.returncode == 0, proc.stderr
    assert any(c.startswith("ec2 start-instances --instance-ids i-0abc") for c in calls)
    assert any(c.startswith("ec2 wait instance-running") for c in calls)
    assert "http://203.0.113.7:5000" in proc.stdout


def test_stop_backs_up_before_stopping(tmp_path):
    proc, calls = run(tmp_path, "stop")
    assert proc.returncode == 0, proc.stderr
    send = next(i for i, c in enumerate(calls) if c.startswith("ssm send-command"))
    stop = next(i for i, c in enumerate(calls) if c.startswith("ec2 stop-instances"))
    assert send < stop
    assert "backup-postgres.sh" in calls[send]


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
