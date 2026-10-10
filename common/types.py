import re
from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
DatasetName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=64)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
# Machine name of a class (classes.code): used by code, the model, manifests and payloads.
ClassCode = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)]
# Machine name of a crop (crops.code), e.g. coffee, durian.
CropCode = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_]*$", max_length=64)]
# Full Git commit SHA of the merge to main that started a round or a release.
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
# Worker image GitHub Actions built for that commit; every job runs in it.
WORKER_IMAGE_REPO = "ghcr.io/duytq5/duytq5-leaf-al-platform-worker"
WorkerImage = Annotated[
    str, StringConstraints(pattern="^" + re.escape(WORKER_IMAGE_REPO) + ":[0-9a-f]{40}$")
]


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
