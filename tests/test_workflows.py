"""The GitHub Actions workflows (design doc "Control through GitHub")."""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"


def _load(name: str) -> dict:
    wf = yaml.safe_load((WORKFLOWS / name).read_text())
    wf["on"] = wf.pop(True)  # YAML 1.1 reads the key `on` as true
    return wf


def _run_lines(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_release_runs_on_merges_to_release_files():
    wf = _load("release.yml")
    assert wf["on"]["push"] == {"branches": ["main"], "paths": ["configs/releases/**"]}
    # Only an OIDC token and a read of the repo; no stored AWS keys.
    assert wf["permissions"] == {"id-token": "write", "contents": "read"}
    # Two quick merges: the newer one waits, nothing runs side by side.
    assert wf["concurrency"] == {"group": "release", "cancel-in-progress": False}
    text = (WORKFLOWS / "release.yml").read_text()
    assert "secrets." not in text


def test_release_job_assumes_the_role_and_waits():
    (job,) = _load("release.yml")["jobs"].values()
    assert job["timeout-minutes"] == 360
    creds = next(s for s in job["steps"] if "configure-aws-credentials" in s.get("uses", ""))
    assert creds["with"]["role-to-assume"] == "${{ vars.AWS_ROLE_ARN }}"
    assert creds["with"]["aws-region"] == "ap-southeast-2"
    run = _run_lines(job)
    assert "al release-input" in run
    assert "stepfunctions start-execution" in run
    assert "stepfunctions describe-execution" in run
    assert "/leaf-al/state-machines/model-release" in run


def test_pinned_actions():
    for path in WORKFLOWS.glob("*.yml"):
        for job in yaml.safe_load(path.read_text())["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    assert "@" in step["uses"], (path.name, step["uses"])
