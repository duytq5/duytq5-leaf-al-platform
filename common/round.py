"""Human-written settings for an AL round: configs/rounds/<name>.yaml in the
ops repo (example: configs/examples/round.yaml).

The owner explores the data, then picks the training recipe and the selection
strategy, their parameters and k for each round. Merging the file to the ops
repo's main starts the round (design doc, "Control through GitHub"). Nothing about
training or selection is hardcoded in the pipeline; the chosen values are
stored on the `rounds` row and model_versions.train_config, so every round
is reproducible.
"""

from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from common.types import DatasetName, WorkerImage

# A registered MLflow model to fine-tune from: a version or an alias,
# e.g. models:/rocole-fastvit/3 or models:/rocole-fastvit@champion.
BaseModelUri = Annotated[
    str,
    StringConstraints(
        pattern=r"^models:/[A-Za-z0-9][A-Za-z0-9._-]*(/[1-9][0-9]*|@[A-Za-z0-9_-]+)$"
    ),
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectionConfig(_Strict):
    strategy: str  # a name registered in selection.STRATEGIES
    params: dict[str, Any] = Field(default_factory=dict)  # validated by the strategy
    k: int = Field(gt=0)
    seed: int = 42


class TrainSection(_Strict):
    recipe: str  # a name registered in recipes (`al recipes`)
    params: dict[str, Any] = Field(default_factory=dict)  # validated by the recipe
    # None: start from the recipe's own weights (ImageNet for timm_classifier).
    base_model: BaseModelUri | None = None


class RoundConfig(_Strict):
    dataset: DatasetName
    # The dataset version to train on; None: the latest.
    base_version: int | None = Field(default=None, ge=0)
    train: TrainSection
    selection: SelectionConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RoundConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class RoundExecution(_Strict):
    """Input of one al-round execution, set by the start-round Lambda.

    Everything else (dataset, base version, strategy, k, seed) is on the
    `rounds` row. `train` is the round config's training section, sent to the
    worker in the train job and stored on model_versions.train_config.
    """

    round_id: UUID
    image: WorkerImage
    train: TrainSection
