from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
DatasetName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=64)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
# Machine name of a class (classes.code): used by code, the model, manifests and payloads.
ClassCode = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)]


class ImageStatus(StrEnum):
    PENDING = "pending"
    UNLABELED = "unlabeled"
    QUEUED = "queued"
    LABELED = "labeled"
    REJECTED = "rejected"  # expert marked it "not a leaf"; terminal, never trained on


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ImageSource(StrEnum):
    SEED = "seed"
    EDGE = "edge"
