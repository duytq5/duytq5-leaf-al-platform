"""Crop and dataset settings read by the seed script.

configs/crops.yaml lists the crops (plants) the app shows. A dataset belongs
to one crop and has one class list and a fixed train/val/test split.

configs/datasets/<name>.yaml is read by the seed script, which stores the
classes in the `classes` table. A class has a machine `code` (used by code,
the model, the manifest and API payloads) and a readable `display_name`. The
order of the list is the model's output order.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    model_config = ConfigDict(extra="forbid")

    val: float = Field(ge=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def _train_left(self) -> Self:
        if self.val + self.test >= 1.0:
            raise ValueError("val + test must leave some images for train")
        return self


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
