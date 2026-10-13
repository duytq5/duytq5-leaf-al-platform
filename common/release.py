"""Model release and rollback (design doc F5).

configs/releases/{dataset}.yaml holds the dataset's current model:

    model: durian-fastvit   # MLflow registered model name
    version: 3              # MLflow model version

Merging a change to it runs the `release` workflow, which starts the
model-release state machine with a ReleaseExecution. Rollback is a PR that sets
an earlier version (usually `git revert`); it runs the same way.
"""

from pathlib import Path
from uuid import UUID, uuid5

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from common.types import CommitSha, DatasetName, GitHubLogin, ModelName, Sha256

RELEASES_DIR = Path("configs/releases")
# Namespace of export job ids (uuid5), fixed forever.
EXPORT_JOB_NAMESPACE = UUID("4f6c1d0e-3b1a-5c55-9a7e-6c2f0b8e1d21")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReleaseFile(_Strict):
    model: ModelName
    version: int = Field(ge=1)

    @classmethod
    def from_yaml(cls, path: str | Path) -> tuple[str, "ReleaseFile"]:
        """The dataset (the file's name) and its release."""
        path = Path(path)
        name = TypeAdapter(DatasetName).validate_python(path.stem)
        with open(path, encoding="utf-8") as f:
            return name, cls.model_validate(yaml.safe_load(f))


class ReleaseExecution(_Strict):
    """Input of one model-release execution, set by the release workflow."""

    dataset: DatasetName
    model: ModelName
    version: int = Field(ge=1)
    released_by: GitHubLogin  # who merged the release PR
    commit_sha: CommitSha


class ExportResult(_Strict):
    """What the worker returns for an export job (SendTaskSuccess)."""

    onnx_s3_key: str
    onnx_sha256: Sha256
    size_bytes: int = Field(gt=0)


def export_job_uuid(ev: ReleaseExecution) -> UUID:
    """The same on every retry of one release. A later release of the same
    version (a rollback) is a new job, so the worker moves the aliases again."""
    return uuid5(EXPORT_JOB_NAMESPACE, f"{ev.model}/{ev.version}/{ev.commit_sha}")
