"""Model manifest stored at edge/{model}/{version}/manifest.json.

The app reads preprocessing and the label list from here instead of hardcoding
them. `labels` is the class snapshot frozen when the model was trained (also
stored as model_versions.labels): one {code, display_name} per output index,
in output order, so the app can show readable names.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.types import ClassCode, Sha256


class Preprocess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class ModelClass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ClassCode
    display_name: str = Field(min_length=1)


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    version: int = Field(ge=1)
    format: Literal["onnx"] = "onnx"
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    preprocess: Preprocess
    labels: list[ModelClass] = Field(min_length=2)  # index = model output index

    @field_validator("labels")
    @classmethod
    def _unique_codes(cls, v: list[ModelClass]) -> list[ModelClass]:
        if len({c.code for c in v}) != len(v):
            raise ValueError("class codes must be unique")
        return v
