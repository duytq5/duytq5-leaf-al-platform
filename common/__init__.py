"""Shared payloads used by the worker, Lambdas, CLI and edge API.

Every message that crosses a process boundary (SQS job, Edge API body, model
manifest, training config) is defined here as a Pydantic model, so producers
and consumers validate against the same schema.
"""

from common.config import TrainConfig
from common.dataset import DatasetConfig, SplitConfig
from common.edge import CaptureRequest, CaptureResponse, CaptureStatus, DeviceConfig
from common.jobs import ExportJob, JobMessage, JobType, ScoreJob, TrainJob, parse_job
from common.manifest import ModelManifest
from common.round import RoundConfig, SelectionConfig
from common.types import ImageSource, ImageStatus, Split

__all__ = [
    "CaptureRequest",
    "CaptureResponse",
    "CaptureStatus",
    "DatasetConfig",
    "DeviceConfig",
    "ExportJob",
    "ImageSource",
    "ImageStatus",
    "JobMessage",
    "JobType",
    "ModelManifest",
    "RoundConfig",
    "ScoreJob",
    "SelectionConfig",
    "Split",
    "SplitConfig",
    "TrainConfig",
    "TrainJob",
    "parse_job",
]
