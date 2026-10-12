"""Edge API request and response bodies.

POST /v1/captures                     -> CaptureRequest / CaptureResponse
GET  /v1/devices/config?dataset={name} -> DeviceConfig
GET  /v1/crops                        -> CropsResponse

The uploader (user_sub) is taken from the Cognito JWT, never from the body.
Retrying a capture with the same capture_id and body must return the same
response.
"""

from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, model_validator

from common.types import ClassCode, CropCode, DatasetName, Probability, Sha256

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PROBS_SUM_TOLERANCE = 0.01


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageInfo(_Strict):
    sha256: Sha256
    content_type: Literal["image/jpeg", "image/png"]
    size: int = Field(gt=0, le=MAX_UPLOAD_BYTES)


class Inference(_Strict):
    """The on-device prediction. The Edge API computes top1 and confidence
    from probs and checks the keys against the model version's classes."""

    model_version: str  # e.g. "leaf-disease/12"
    probs: dict[ClassCode, Probability] = Field(min_length=2)

    @model_validator(mode="after")
    def _probs_sum_to_one(self) -> Self:
        if abs(sum(self.probs.values()) - 1) > PROBS_SUM_TOLERANCE:
            raise ValueError(f"probs must sum to 1 (±{PROBS_SUM_TOLERANCE})")
        return self


class CaptureRequest(_Strict):
    capture_id: UUID
    dataset: DatasetName
    captured_at: AwareDatetime
    image: ImageInfo
    inference: Inference


class CaptureStatus(StrEnum):
    UPLOAD_REQUIRED = "upload_required"
    ALREADY_EXISTS = "already_exists"


class UploadInstruction(_Strict):
    url: HttpUrl
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str] = {}  # e.g. Content-Type; the PUT must send them
    expires_in: int = Field(gt=0)


class CaptureResponse(_Strict):
    capture_id: UUID
    status: CaptureStatus
    upload: UploadInstruction | None = None

    @model_validator(mode="after")
    def _upload_matches_status(self) -> Self:
        needs_upload = self.status is CaptureStatus.UPLOAD_REQUIRED
        if needs_upload != (self.upload is not None):
            raise ValueError("upload is required exactly when status is upload_required")
        return self


class DeviceConfig(_Strict):
    """Current champion model of one dataset and where to download it."""

    dataset: DatasetName
    model: str
    version: int = Field(ge=1)
    sha256: Sha256  # of the ONNX file
    manifest_url: HttpUrl
    model_url: HttpUrl
    expires_in: int = Field(gt=0)


class CropDataset(_Strict):
    """A dataset (model) of a crop that has a released model."""

    name: DatasetName  # pass to GET /v1/devices/config?dataset=
    display_name: str
    version: int = Field(ge=1)  # current release; the app always gets this one


class Crop(_Strict):
    code: CropCode
    display_name: str
    description: str | None = None
    datasets: list[CropDataset] = Field(min_length=1)  # crops without a release are left out


class CropsResponse(_Strict):
    """Public list of crops and their released datasets, sorted by display_name."""

    crops: list[Crop]
