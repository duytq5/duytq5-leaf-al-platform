"""Selection strategy registry.

Each strategy declares the score outputs it needs (`requires`), so the score
job computes only those, and a `Params` model for the settings a human can
choose per round. Adding a method is one new registered class; the pipeline
does not change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from common.round import SelectionConfig

ScoreOutput = Literal["probs", "embeddings"]


class StrategyParams(BaseModel):
    """Base for a strategy's per-round parameters. Unknown keys are rejected, so a typo
    in a round config fails instead of silently using a default."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class PoolData:
    """Unlabeled pool images plus their score outputs, row-aligned with `ids`."""

    ids: list[str]
    probs: np.ndarray | None = None  # (n, num_classes)
    embeddings: np.ndarray | None = None  # (n, dim)

    def __post_init__(self) -> None:
        n = len(self.ids)
        for name in ("probs", "embeddings"):
            arr = getattr(self, name)
            if arr is not None and (arr.ndim != 2 or arr.shape[0] != n):
                raise ValueError(f"{name} must have shape ({n}, d), got {arr.shape}")

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class LabeledStats:
    """Aggregate statistics of the labeled set, not per-image data.

    Loaded with one GROUP BY over the labels table (see LABELED_COUNTS_SQL), so
    a round never pulls labeled-image metadata into memory.
    """

    class_counts: np.ndarray  # (num_classes,) labeled images per class, label-list order

    def __post_init__(self) -> None:
        if self.class_counts.ndim != 1 or (self.class_counts < 0).any():
            raise ValueError("class_counts must be a 1-D array of non-negative counts")

    @property
    def total(self) -> int:
        return int(self.class_counts.sum())

    @classmethod
    def empty(cls, num_classes: int) -> "LabeledStats":
        return cls(class_counts=np.zeros(num_classes, dtype=np.int64))

    @classmethod
    def from_counts(cls, counts: dict[str, int], label_list: list[str]) -> "LabeledStats":
        """Map `{label: count}` rows from SQL onto the dataset's label-list order."""
        unknown = set(counts) - set(label_list)
        if unknown:
            raise ValueError(f"labels not in the dataset label list: {sorted(unknown)}")
        return cls(class_counts=np.array([counts.get(y, 0) for y in label_list], dtype=np.int64))


# Per-class labeled counts for one dataset's training split. Test-split images
# never enter the pool, so they are excluded here too.
LABELED_COUNTS_SQL = """
SELECT l.label, count(DISTINCT l.image_id) AS n
FROM labels l
JOIN images i ON i.id = l.image_id
WHERE i.dataset = %(dataset)s AND i.split = 'train'
GROUP BY l.label
"""


class Strategy(ABC):
    requires: ClassVar[frozenset[ScoreOutput]] = frozenset()
    Params: ClassVar[type[StrategyParams]] = StrategyParams
    description: ClassVar[str] = ""

    def __init__(self, seed: int = 0, params: dict[str, Any] | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.params = self.Params.model_validate(params or {})

    @abstractmethod
    def select(self, pool: PoolData, labeled: LabeledStats, k: int) -> list[str]:
        """Return the ids of up to k images from `pool` to label next."""


STRATEGIES: dict[str, type[Strategy]] = {}


def register(name: str):
    def deco(cls: type[Strategy]) -> type[Strategy]:
        if name in STRATEGIES:
            raise ValueError(f"strategy {name!r} is already registered")
        if not issubclass(cls.Params, StrategyParams):
            raise TypeError(f"{cls.__name__}.Params must subclass StrategyParams")
        STRATEGIES[name] = cls
        return cls

    return deco


def get_strategy(name: str, seed: int = 0, params: dict[str, Any] | None = None) -> Strategy:
    """Build a strategy; raises KeyError for an unknown name, ValidationError for bad params."""
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(STRATEGIES)}") from None
    return cls(seed=seed, params=params)


def from_config(cfg: SelectionConfig) -> Strategy:
    return get_strategy(cfg.strategy, seed=cfg.seed, params=cfg.params)


def describe_strategies() -> dict[str, dict[str, Any]]:
    """What a human needs to pick a strategy: description, required inputs, parameters."""
    return {
        name: {
            "description": cls.description,
            "requires": sorted(cls.requires),
            "params": cls.Params.model_json_schema().get("properties", {}),
        }
        for name, cls in sorted(STRATEGIES.items())
    }
