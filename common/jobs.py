"""SQS job messages sent by Step Functions to the worker.

{"job_id": uuid, "type": "train | score | export", "image": str,
 "task_token": str, "round_id": uuid | null, "spec": {...}}

train and score come from the al-round state machine, export from the
model-release state machine; all three carry a task token. `image` is the
worker image built for the commit that started the round or release; the
worker runs the job in it.

SQS can deliver a message twice, so job handlers must be idempotent on job_id.
"""

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from common.config import TrainConfig
from common.manifest import ClassList
from common.types import WorkerImage


class JobType(StrEnum):
    TRAIN = "train"
    SCORE = "score"
    EXPORT = "export"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainSpec(_Strict):
    config: TrainConfig
    manifest_uri: str
    # Output layer order; the round's Lambda stores the same list in
    # model_versions.labels.
    classes: ClassList


class ScoreSpec(_Strict):
    model_uri: str
    pool_manifest_uri: str
    outputs: frozenset[Literal["probs", "embeddings"]] = frozenset({"probs"})
    output_uri: str

    @field_validator("outputs")
    @classmethod
    def _probs_required(cls, v: frozenset[str]) -> frozenset[str]:
        if "probs" not in v:
            raise ValueError("score outputs must include 'probs'")
        return v


class ExportSpec(_Strict):
    model_uri: str
    output_prefix: str
    labels: ClassList  # copied from model_versions.labels into manifest.json as is


class _JobBase(_Strict):
    job_id: UUID
    image: WorkerImage
    task_token: str


class TrainJob(_JobBase):
    type: Literal[JobType.TRAIN] = JobType.TRAIN
    round_id: UUID
    spec: TrainSpec


class ScoreJob(_JobBase):
    type: Literal[JobType.SCORE] = JobType.SCORE
    round_id: UUID
    spec: ScoreSpec


class ExportJob(_JobBase):
    """Sent by the model-release state machine; a release has no round."""

    type: Literal[JobType.EXPORT] = JobType.EXPORT
    round_id: None = None
    spec: ExportSpec


JobMessage = Annotated[TrainJob | ScoreJob | ExportJob, Field(discriminator="type")]

_adapter: TypeAdapter[JobMessage] = TypeAdapter(JobMessage)


def parse_job(body: str | bytes) -> TrainJob | ScoreJob | ExportJob:
    """Parse and validate an SQS message body."""
    return _adapter.validate_json(body)
