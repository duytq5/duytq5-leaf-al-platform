"""The platform repo's GitHub Actions workflows."""

from pathlib import Path

import yaml

from common.types import WORKER_IMAGE_REPO

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"


def _load(name: str) -> dict:
    wf = yaml.safe_load((WORKFLOWS / name).read_text())
    wf["on"] = wf.pop(True)  # YAML 1.1 reads the key `on` as true
    return wf


def test_worker_image_is_built_for_every_merge_to_main():
    """The ops repo pins a platform commit and its rounds run in that
    commit's image, so every commit on main gets one. A PR that changes the
    image's inputs builds it too, without pushing."""
    wf = _load("worker-image.yml")
    assert wf["on"]["push"] == {"branches": ["main"]}
    assert "workflow_dispatch" in wf["on"]
    assert set(wf["on"]["pull_request"]["paths"]) == {
        "worker/Dockerfile",
        ".dockerignore",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/worker-image.yml",
    }
    # Push to GHCR with the workflow's own token; no AWS, no stored secrets.
    assert wf["permissions"] == {"contents": "read", "packages": "write"}
    assert "secrets." not in (WORKFLOWS / "worker-image.yml").read_text()


def test_worker_image_is_tagged_with_the_commit():
    (job,) = _load("worker-image.yml")["jobs"].values()
    build = next(s for s in job["steps"] if "build-push-action" in s.get("uses", ""))
    assert build["with"]["file"] == "worker/Dockerfile"
    # PR code never pushes and never logs in.
    assert build["with"]["push"] == "${{ github.event_name != 'pull_request' }}"
    assert build["with"]["tags"] == WORKER_IMAGE_REPO + ":${{ github.sha }}"
    # The image knows its commit; MLflow records it as git_sha.
    assert build["with"]["build-args"] == "GIT_SHA=${{ github.sha }}"
    login = next(s for s in job["steps"] if "login-action" in s.get("uses", ""))
    assert login["with"]["registry"] == "ghcr.io"
    assert login["with"]["password"] == "${{ github.token }}"
    assert login["if"] == "github.event_name != 'pull_request'"
