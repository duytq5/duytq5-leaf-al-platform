"""Crop and dataset settings read by the seed script.

configs/crops.yaml lists the crops (plants) the app shows. A dataset belongs
to one crop and has one class list and a fixed train/val/test split.

configs/datasets/<name>.yaml is read by the seed script, which stores the
classes in the `classes` table. A class has a machine `code` (used by code,
the model, the manifest and API payloads) and a readable `display_name`. The
order of the list is the model's output order.
"""

import re
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.types import ClassCode, CropCode, DatasetName


class CropConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: CropCode
    display_name: str = Field(min_length=1)
    description: str | None = None


class CropsConfig(BaseModel):
    """configs/crops.yaml. The seed upserts every crop by code."""

    model_config = ConfigDict(extra="forbid")

    crops: list[CropConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_codes(self) -> Self:
        if len({c.code for c in self.crops}) != len(self.crops):
            raise ValueError("crop codes must be unique")
        return self

    @property
    def codes(self) -> set[str]:
        return {c.code for c in self.crops}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CropsConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class SplitConfig(BaseModel):
    """How the seed splits a dataset. Fixed once an image is seeded.

    group_pattern: a regex on the file name; images with the same match (the
    first capture group if there is one) always land in the same split, e.g.
    the views of one plant. Without it, every image is its own group.
    initial_labeled: share of each class's train images labeled at seeding.
    They go into v0, so the first round has something to train on.
    """

    model_config = ConfigDict(extra="forbid")

    val: float = Field(ge=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)
    seed: int = 42
    group_pattern: str | None = None
    initial_labeled: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _train_left(self) -> Self:
        if self.val + self.test >= 1.0:
            raise ValueError("val + test must leave some images for train")
        return self

    @field_validator("group_pattern")
    @classmethod
    def _compiles(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"group_pattern is not a valid regex: {e}") from e
        return v


class ClassConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ClassCode
    display_name: str = Field(min_length=1)
    description: str | None = None


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: DatasetName
    crop: CropCode
    display_name: str = Field(min_length=1)
    description: str | None = None
    classes: list[ClassConfig] = Field(min_length=2)  # in model output order
    split: SplitConfig

    @model_validator(mode="after")
    def _unique_codes(self) -> Self:
        if len(set(self.codes)) != len(self.codes):
            raise ValueError("class codes must be unique")
        return self

    @property
    def codes(self) -> list[str]:
        return [c.code for c in self.classes]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))
