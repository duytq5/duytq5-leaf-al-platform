"""deploy/ec2/bootstrap.sh with stub aws, docker and crontab (no AWS, no Docker)."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = "a" * 48  # write-env.sh refuses Postgres passwords under 32 characters

STUBS = {
    "aws": f'echo "$*" >> "$CALLS"; echo {SECRET}\n',
    "docker": 'echo "docker $*" >> "$CALLS"\n',
    # crontab -l prints the saved table; crontab - saves stdin.
    "crontab": 'if [ "$1" = "-l" ]; then cat "$CRON" 2>/dev/null; else cat > "$CRON"; fi\n',
}


def run(tmp_path):
    repo = tmp_path / "home" / "duytq5-leaf-al-platform"
    shutil.copytree(
        ROOT / "deploy" / "ec2",
        repo / "deploy" / "ec2",
        dirs_exist_ok=True,
        # CI runs the real compose Postgres first: skip its root-owned runtime files.
        ignore=shutil.ignore_patterns("data", "tls", ".env"),
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        f = bin_dir / name
        f.write_text("#!/bin/sh\n" + body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "AWS_REGION": "ap-southeast-2",
        "CALLS": str(tmp_path / "calls.log"),
        "CRON": str(tmp_path / "crontab"),
    }
    proc = subprocess.run(
        ["sh", str(repo / "deploy" / "ec2" / "bootstrap.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, repo, tmp_path / "crontab"


def test_writes_env_starts_compose_and_installs_one_cron_line(tmp_path):
    (tmp_path / "crontab").write_text("5 * * * * other-job\n")
    for _ in range(2):  # idempotent
        proc, repo, cron = run(tmp_path)
        assert proc.returncode == 0, proc.stderr
    env_file = repo / "deploy" / "ec2" / ".env"
    assert f"POSTGRES_PASSWORD={SECRET}" in env_file.read_text()
    assert "docker compose up -d --build" in (tmp_path / "calls.log").read_text()
    lines = cron.read_text().splitlines()
    assert "5 * * * * other-job" in lines
    backups = [line for line in lines if "backup-postgres.sh" in line]
    assert backups == [
        f"0 3 * * * {repo}/deploy/ec2/backup-postgres.sh >> {repo.parent}/leaf-al-backup.log 2>&1"
    ]
