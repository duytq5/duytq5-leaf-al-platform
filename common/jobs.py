"""SQS job messages sent by Step Functions (or the promote CLI) to the worker.

{"job_id": uuid, "type": "train | score | export", "task_token": str,
 "round_id": uuid, "spec": {...}}

SQS can deliver a message twice, so job handlers must be idempotent on job_id.
"""

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from common.config import TrainConfig


class JobType(StrEnum):
    TRAIN = "train"
    SCORE = "score"
    EXPORT = "export"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainSpec(_Strict):
    config: TrainConfig
    manifest_uri: str


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


class _JobBase(_Strict):
    job_id: UUID


class TrainJob(_JobBase):
    type: Literal[JobType.TRAIN] = JobType.TRAIN
    task_token: str
    round_id: UUID
    spec: TrainSpec


class ScoreJob(_JobBase):
    type: Literal[JobType.SCORE] = JobType.SCORE
    task_token: str
    round_id: UUID
    spec: ScoreSpec


class ExportJob(_JobBase):
    """Run by the promote CLI, which may not go through Step Functions."""

    type: Literal[JobType.EXPORT] = JobType.EXPORT
    task_token: str | None = None
    round_id: UUID | None = None
    spec: ExportSpec


JobMessage = Annotated[TrainJob | ScoreJob | ExportJob, Field(discriminator="type")]

_adapter: TypeAdapter[JobMessage] = TypeAdapter(JobMessage)


def parse_job(body: str | bytes) -> TrainJob | ScoreJob | ExportJob:
    """Parse and validate an SQS message body."""
    return _adapter.validate_json(body)
