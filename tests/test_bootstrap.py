"""deploy/ec2/bootstrap.sh with stub aws, docker and crontab (no AWS, no Docker)."""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = "a" * 48  # write-env.sh refuses Postgres passwords under 32 characters

LABEL_STUDIO = {
    "admin_email": "owner@example.com",
    "admin_password": "p" * 32,
    "api_token": "t" * 40,
    "webhook_secret": "w" * 40,
    "cookie_secret": "c" * 32,
}

STUBS = {
    # Secrets Manager returns the Label Studio JSON, everything else a string.
    "aws": 'echo "$*" >> "$CALLS"\n'
    f"case \"$1\" in secretsmanager) echo '{json.dumps(LABEL_STUDIO)}' ;;"
    f" *) echo {SECRET} ;; esac\n",
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
    calls = (tmp_path / "calls.log").read_text()
    assert calls.count("docker compose up -d --build") == 2
    assert calls.count("docker compose restart postgres") == 1  # same certificate the 2nd time
    lines = cron.read_text().splitlines()
    assert "5 * * * * other-job" in lines
    backups = [line for line in lines if "backup-postgres.sh" in line]
    assert backups == [
        f"0 3 * * * {repo}/deploy/ec2/backup-postgres.sh >> {repo.parent}/leaf-al-backup.log 2>&1"
    ]


def test_label_studio_settings_and_database(tmp_path):
    proc, repo, _ = run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    env = dict(
        line.split("=", 1)
        for line in (repo / "deploy" / "ec2" / ".env").read_text().splitlines()
        if "=" in line
    )
    assert env["LS_DOMAIN"] == SECRET  # the stub answers every SSM read with SECRET
    assert env["VIEWER_DOMAIN"] == SECRET
    assert env["MLFLOW_DOMAIN"] == SECRET
    assert env["LS_ADMIN_EMAIL"] == "owner@example.com"
    assert env["LS_ADMIN_PASSWORD"] == LABEL_STUDIO["admin_password"]
    assert env["LS_API_TOKEN"] == LABEL_STUDIO["api_token"]
    assert env["OAUTH2_PROXY_COOKIE_SECRET"] == LABEL_STUDIO["cookie_secret"]
    assert env["OAUTH2_PROXY_CLIENT_SECRET"] == SECRET
    assert env["OAUTH2_PROXY_OIDC_ISSUER_URL"] == (
        f"https://cognito-idp.ap-southeast-2.amazonaws.com/{SECRET}"
    )
    calls = (tmp_path / "calls.log").read_text()
    assert (
        "secretsmanager get-secret-value --region ap-southeast-2 --secret-id leaf-al/label-studio"
        in calls
    )
    assert "cognito-idp describe-user-pool-client" in calls
    assert "--name /leaf-al/viewer/domain" in calls
    assert "--name /leaf-al/mlflow/domain" in calls
    # Its database exists before Label Studio starts.
    create = calls.index("CREATE DATABASE labelstudio")
    assert calls.index("docker compose up -d --wait postgres") < create
    assert create < calls.index("docker compose up -d --build")


def test_write_env_refuses_a_partial_label_studio_secret(tmp_path):
    partial = {k: v for k, v in LABEL_STUDIO.items() if k != "webhook_secret"}
    STUBS_SAVED = dict(STUBS)
    try:
        STUBS["aws"] = STUBS["aws"].replace(json.dumps(LABEL_STUDIO), json.dumps(partial))
        proc, _, _ = run(tmp_path)
    finally:
        STUBS.update(STUBS_SAVED)
    assert proc.returncode != 0
    assert "webhook_secret" in proc.stderr


def test_caddy_restarts_only_when_its_caddyfile_changed(tmp_path):
    """Compose does not recreate Caddy for a new Caddyfile (a bind mount, and
    git replaces the file), and Caddy reads it only at start."""
    for _ in range(2):
        proc, repo, _ = run(tmp_path)
        assert proc.returncode == 0, proc.stderr
    calls = (tmp_path / "calls.log").read_text()
    assert calls.count("docker compose restart caddy") == 1  # same file the 2nd time
    assert calls.index("docker compose up -d --build") < calls.index("docker compose restart caddy")

    caddyfile = repo / "deploy" / "ec2" / "caddy" / "Caddyfile"
    caddyfile.write_text(caddyfile.read_text() + "\n# changed\n")
    (tmp_path / "calls.log").write_text("")
    proc = subprocess.run(
        ["sh", str(repo / "deploy" / "ec2" / "bootstrap.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "AWS_REGION": "ap-southeast-2",
            "CALLS": str(tmp_path / "calls.log"),
            "CRON": str(tmp_path / "crontab"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "docker compose restart caddy" in (tmp_path / "calls.log").read_text()
