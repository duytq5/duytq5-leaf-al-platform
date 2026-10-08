"""Edge API request and response bodies.

POST /v1/captures     -> CaptureRequest / CaptureResponse
GET  /v1/devices/config -> DeviceConfig

device_id is taken from the Cognito JWT, never from the body. Retrying a
capture with the same capture_id must return the same response.
"""

from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, model_validator

from common.manifest import ModelManifest
from common.types import ClassCode, DatasetName, Probability, Sha256

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageInfo(_Strict):
    sha256: Sha256
    content_type: Literal["image/jpeg"]
    size: int = Field(gt=0, le=MAX_UPLOAD_BYTES)


class Inference(_Strict):
    model_version: str  # e.g. "leaf-disease/12"
    top1: str
    confidence: Probability
    probs: dict[str, Probability]

    @model_validator(mode="after")
    def _top1_in_probs(self) -> Self:
        if self.top1 not in self.probs:
            raise ValueError("top1 must be one of the keys in probs")
        return self


class Feedback(_Strict):
    """The user's optional opinion of the prediction.

    Shown to the expert as a hint; never stored as a label.
    """

    agrees: bool | None = None
    suggested_class: ClassCode | None = None

    @model_validator(mode="after")
    def _suggest_only_on_disagree(self) -> Self:
        if self.suggested_class is not None and self.agrees is not False:
            raise ValueError("suggested_class is only allowed when agrees is false")
        return self


class CaptureRequest(_Strict):
    capture_id: UUID
    dataset: DatasetName
    captured_at: AwareDatetime
    image: ImageInfo
    inference: Inference
    feedback: Feedback | None = None


class CaptureStatus(StrEnum):
    UPLOAD_REQUIRED = "upload_required"
    ALREADY_EXISTS = "already_exists"


class UploadInstruction(_Strict):
    url: HttpUrl
    method: Literal["PUT"] = "PUT"
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
    """Current champion model and where to download it."""

    model: str
    version: int = Field(ge=1)
    download_url: HttpUrl
    expires_in: int = Field(gt=0)
    manifest: ModelManifest
