"""Admin and labeler read API bodies (design doc F3; read only: rounds and
releases are started through GitHub).

GET /v1/rounds/active                 -> ActiveRoundsResponse   (labelers, admins)
GET /v1/rounds/{round_id}             -> RoundStatus            (admins)
GET /v1/datasets/{name}/versions      -> DatasetVersionsResponse (admins)
GET /v1/datasets/{name}/versions/{n}  -> VersionItemsResponse    (admins)
"""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from common.types import ClassCode, DatasetName, Split


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoundPhase(StrEnum):
    """Where an unfinished round is, from Postgres and its selection file."""

    STARTING = "starting"  # pending: the state machine has not marked it running
    TRAINING = "training"  # snapshot, train, score, select: no selection yet
    AWAITING_LABELS = "awaiting_labels"  # the expert labels the selection
    MERGING = "merging"  # every selected image is labeled or rejected


RoundStatusName = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class ActiveRound(_Strict):
    round_id: UUID
    dataset: DatasetName
    phase: RoundPhase
    k: int = Field(gt=0)
    # Selected images the expert labeled or rejected.
    labeled: int = Field(ge=0)
    # The round's Label Studio project, once the label step made it.
    label_studio_url: str | None


class ActiveRoundsResponse(_Strict):
    rounds: list[ActiveRound]


class RoundStatus(_Strict):
    round_id: UUID
    dataset: DatasetName
    status: RoundStatusName
    phase: RoundPhase | None  # None once the round finished
    strategy: str
    k: int = Field(gt=0)
    labeled: int = Field(ge=0)
    base_version: int = Field(ge=0)
    started_by: str
    commit_sha: str
    # The training run in MLflow, once the round has a model version.
    mlflow_run_url: str | None
    created_at: AwareDatetime
    finished_at: AwareDatetime | None


class DatasetVersion(_Strict):
    version: int = Field(ge=0)
    parent_version: int | None
    created_by_round: UUID | None
    created_at: AwareDatetime
    total: int = Field(ge=0)
    label_counts: dict[ClassCode, int]  # every class of the dataset, 0 included


class DatasetVersionsResponse(_Strict):
    dataset: DatasetName
    versions: list[DatasetVersion]  # newest first


class VersionItem(_Strict):
    image_id: int
    s3_key: str
    split: Split
    class_code: ClassCode
    label_source: Literal["expert", "oracle", "seed"]
    round_id: UUID | None


class VersionItemsResponse(_Strict):
    dataset: DatasetName
    version: int = Field(ge=0)
    items: list[VersionItem]  # by image_id
    # Pass as ?cursor= for the next page; None on the last page.
    next_cursor: str | None
