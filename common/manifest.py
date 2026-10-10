"""Model manifest stored at edge/{model}/{version}/manifest.json.

The app reads preprocessing and the label list from here instead of hardcoding
them. `labels` is the class snapshot frozen when the model was trained (also
stored as model_versions.labels): one {code, display_name} per output index,
in output order, so the app can show readable names.
"""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

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


def _unique_codes(v: list[ModelClass]) -> list[ModelClass]:
    if len({c.code for c in v}) != len(v):
        raise ValueError("class codes must be unique")
    return v


# The class snapshot: one entry per model output index, in output order.
ClassList = Annotated[list[ModelClass], Field(min_length=2), AfterValidator(_unique_codes)]


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    version: int = Field(ge=1)
    format: Literal["onnx"] = "onnx"
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    preprocess: Preprocess
    labels: ClassList  # index = model output index
