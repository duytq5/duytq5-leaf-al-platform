"""Training config, loaded from YAML so a new backbone needs no code change."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from common.types import DatasetName


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSection(_Strict):
    arch: str
    source: Literal["timm"] = "timm"
    pretrained: str | None = "imagenet"


class HeadSection(_Strict):
    num_classes: int = Field(ge=2)


class DataSection(_Strict):
    dataset: DatasetName
    version: Annotated[str, StringConstraints(pattern=r"^v\d+$")]


class TrainSection(_Strict):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    # PyYAML reads "3e-4" as a string; lax mode coerces it to float.
    lr: float = Field(gt=0)
    seed: int = 42


class TrainConfig(_Strict):
    model: ModelSection
    base_model: str | None = None  # e.g. "models:/leaf-disease@champion"
    head: HeadSection
    data: DataSection
    train: TrainSection

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))
