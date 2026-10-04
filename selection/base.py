"""Selection strategy registry.

Each strategy declares the score outputs it needs (`requires`), so the score
job computes only those. Adding a method is one new registered class; the
pipeline does not change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

ScoreOutput = Literal["probs", "embeddings"]


@dataclass(frozen=True)
class PoolData:
    """Images plus the score outputs for them, row-aligned with `ids`."""

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


class Strategy(ABC):
    requires: ClassVar[frozenset[ScoreOutput]] = frozenset()

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select(self, pool: PoolData, labeled: PoolData, k: int, **cfg) -> list[str]:
        """Return the ids of up to k images from `pool` to label next."""


STRATEGIES: dict[str, type[Strategy]] = {}


def register(name: str):
    def deco(cls: type[Strategy]) -> type[Strategy]:
        if name in STRATEGIES:
            raise ValueError(f"strategy {name!r} is already registered")
        STRATEGIES[name] = cls
        return cls

    return deco


def get_strategy(name: str, seed: int = 0) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(STRATEGIES)}") from None
    return cls(seed=seed)
