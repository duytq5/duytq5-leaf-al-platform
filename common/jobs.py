"""SQS job messages sent by Step Functions to the worker.

{"job_id": uuid, "type": "train | score | export", "image": str,
 "task_token": str, "round_id": uuid | null, "spec": {...}}

train and score come from the al-round state machine, export from the
model-release state machine; all three carry a task token. `image` is the
worker image built for the commit that started the round or release; the
worker runs the job in it.

SQS can deliver a message twice, so job handlers must be idempotent on job_id.

Parquet files exchanged with the worker (written with pyarrow):
  manifests/{dataset}/v{n}.parquet         image_id, sha256, key, split, class_code
  scores/{round_id}/pool-manifest.parquet  image_id, sha256, key
  scores/{round_id}/pool.parquet           image_id, probs (list<float>, class order)
"""

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from common.manifest import ClassList
from common.round import TrainSection
from common.types import DatasetName, WorkerImage


def job_uuid(round_id: UUID, job_type: "JobType") -> UUID:
    """The job id of a round's train or score job: the same on every retry."""
    return uuid5(round_id, str(job_type))


class JobType(StrEnum):
    TRAIN = "train"
    SCORE = "score"
    EXPORT = "export"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainSpec(TrainSection):
    """The round config's train section (recipe, params, base_model) plus
    what the snapshot step adds."""

    dataset: DatasetName  # names the registered model, with the recipe
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


class TrainResult(_Strict):
    """What the worker returns with SendTaskSuccess for a train job. The round's
    merge step stores it in model_versions."""

    mlflow_name: str
    mlflow_version: int = Field(ge=1)
    mlflow_run_id: str
    model_uri: str  # what the score and export jobs load, e.g. models:/name/3
    arch: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ScoreResult(_Strict):
    output_uri: str
    n_images: int = Field(ge=0)


JobMessage = Annotated[TrainJob | ScoreJob | ExportJob, Field(discriminator="type")]

_adapter: TypeAdapter[JobMessage] = TypeAdapter(JobMessage)


def parse_job(body: str | bytes) -> TrainJob | ScoreJob | ExportJob:
    """Parse and validate an SQS message body."""
    return _adapter.validate_json(body)
