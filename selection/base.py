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
    """Images plus the score outputs for them, row-aligned with `ids`.

    `labels` holds class indices (columns of `probs`, in the dataset's label
    list order) and is set for the labeled set only.
    """

    ids: list[str]
    probs: np.ndarray | None = None  # (n, num_classes)
    embeddings: np.ndarray | None = None  # (n, dim)
    labels: np.ndarray | None = None  # (n,) int class indices

    def __post_init__(self) -> None:
        n = len(self.ids)
        for name in ("probs", "embeddings"):
            arr = getattr(self, name)
            if arr is not None and (arr.ndim != 2 or arr.shape[0] != n):
                raise ValueError(f"{name} must have shape ({n}, d), got {arr.shape}")
        if self.labels is not None and self.labels.shape != (n,):
            raise ValueError(f"labels must have shape ({n},), got {self.labels.shape}")

    def __len__(self) -> int:
        return len(self.ids)


class Strategy(ABC):
    requires: ClassVar[frozenset[ScoreOutput]] = frozenset()
    Params: ClassVar[type[StrategyParams]] = StrategyParams
    description: ClassVar[str] = ""

    def __init__(self, seed: int = 0, params: dict[str, Any] | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.params = self.Params.model_validate(params or {})

    @abstractmethod
    def select(self, pool: PoolData, labeled: PoolData, k: int) -> list[str]:
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
