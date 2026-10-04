"""Model manifest stored at edge/{model}/{version}/manifest.json.

The app reads preprocessing and the label list from here instead of hardcoding
them. `labels` is copied from the dataset's single label list; its order is the
model's output order.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.types import Sha256


class Preprocess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    version: int = Field(ge=1)
    format: Literal["onnx"] = "onnx"
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    preprocess: Preprocess
    labels: list[str] = Field(min_length=2)

    @field_validator("labels")
    @classmethod
    def _unique_labels(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("labels must be unique")
        return v
