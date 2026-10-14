"""Human-written settings for an AL round: configs/rounds/<name>.yaml in the
ops repo (example: configs/examples/round.yaml).

The owner explores the data, then picks the training recipe and the selection
strategy, their parameters and k for each round. A round without `train` only
selects (design doc change 24): it scores the pool with `model`, or with the
dataset's newest model, and training runs later as its own job. A strategy that
needs no scores (random) scores only when `model` is given; otherwise the round
needs no model and no worker. Merging the file to the ops
repo's main starts the round (design doc, "Control through GitHub"). Nothing about
training or selection is hardcoded in the pipeline; the chosen values are
stored on the `rounds` row and model_versions.train_config, so every round
is reproducible.
"""

from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from common.types import DatasetName, WorkerImage

# A registered MLflow model (to fine-tune from, or to score with): a version
# or an alias, e.g. models:/rocole-fastvit/3 or models:/rocole-fastvit@champion.
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
    # None: a selection-only round, scored with `model`.
    train: TrainSection | None = None
    # Only without `train`; None: the dataset's newest model_versions row.
    model: BaseModelUri | None = None
    selection: SelectionConfig

    @model_validator(mode="after")
    def _model_only_without_train(self) -> "RoundConfig":
        if self.train is not None and self.model is not None:
            raise ValueError("a round that trains scores with its own model; drop `model`")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RoundConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class RoundExecution(_Strict):
    """Input of one al-round execution, set by the start-round Lambda.

    Everything else (dataset, base version, strategy, k, seed) is on the
    `rounds` row. `train` is the round config's training section, sent to the
    worker in the train job and stored on model_versions.train_config. A
    selection-only round has no `train` and scores with `model` instead.
    """

    round_id: UUID
    image: WorkerImage
    train: TrainSection | None = None
    model: BaseModelUri | None = None

    @model_validator(mode="after")
    def _train_or_model(self) -> "RoundExecution":
        # Neither: a random round that needs no scores, so no model and no worker.
        if self.train is not None and self.model is not None:
            raise ValueError("a round execution has `train` or `model`, not both")
        return self
