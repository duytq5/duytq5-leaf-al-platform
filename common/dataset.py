"""Per-dataset settings: the one label list and the fixed train/val/test split.

configs/datasets/<name>.yaml is read by the seed script. The label list is
stored on the `datasets` row and its order is the model's output order.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.types import DatasetName


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val: float = Field(ge=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def _train_left(self) -> Self:
        if self.val + self.test >= 1.0:
            raise ValueError("val + test must leave some images for train")
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: DatasetName
    labels: list[str] = Field(min_length=2)
    split: SplitConfig

    @model_validator(mode="after")
    def _unique_labels(self) -> Self:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))
