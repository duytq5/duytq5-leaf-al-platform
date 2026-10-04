"""Uncertainty sampling on predicted class probabilities.

Each strategy turns probs into an uncertainty score (higher = more uncertain)
and takes the top k. Ties keep pool order, so a selection is reproducible.
"""

import numpy as np

from selection.base import PoolData, Strategy, register


def _require_probs(pool: PoolData) -> np.ndarray:
    if pool.probs is None:
        raise ValueError("this strategy needs probs in the pool scores")
    return pool.probs


def top_k(pool: PoolData, scores: np.ndarray, k: int) -> list[str]:
    order = np.argsort(-scores, kind="stable")[: min(k, len(pool))]
    return [pool.ids[i] for i in order]


@register("least_confidence")
class LeastConfidence(Strategy):
    description = "Lowest top-1 probability first (1 - max p)."
    requires = frozenset({"probs"})

    def select(self, pool: PoolData, labeled: PoolData, k: int) -> list[str]:
        probs = _require_probs(pool)
        return top_k(pool, 1.0 - probs.max(axis=1), k)


@register("margin")
class Margin(Strategy):
    description = "Smallest gap between the top two class probabilities first."
    requires = frozenset({"probs"})

    def select(self, pool: PoolData, labeled: PoolData, k: int) -> list[str]:
        probs = _require_probs(pool)
        top2 = np.sort(probs, axis=1)[:, -2:]
        return top_k(pool, -(top2[:, 1] - top2[:, 0]), k)


@register("entropy")
class Entropy(Strategy):
    description = "Highest predictive entropy first."
    requires = frozenset({"probs"})

    def select(self, pool: PoolData, labeled: PoolData, k: int) -> list[str]:
        probs = _require_probs(pool)
        p = np.clip(probs, 1e-12, 1.0)
        return top_k(pool, -(p * np.log(p)).sum(axis=1), k)
