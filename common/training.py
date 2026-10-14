"""A standalone training (design doc change 24): configs/trainings/<name>.yaml
in the ops repo (example: configs/examples/training.yaml).

Rounds may run without training (common/round.py, no `train` section); a
training file trains a model on a dataset version later, with the same recipe
and worker job as a round. Merging the file to the ops repo's main starts it.
"""

from pathlib import Path
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field

from common.round import TrainSection
from common.types import DatasetName, WorkerImage


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainingConfig(_Strict):
    dataset: DatasetName
    # The dataset version to train on; None: the latest.
    version: int | None = Field(default=None, ge=0)
    train: TrainSection

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class TrainingExecution(_Strict):
    """Input of one al-train execution, set by the start-training Lambda; the
    rest (dataset, version, train section) is on the `trainings` row."""

    training_id: UUID
    image: WorkerImage
