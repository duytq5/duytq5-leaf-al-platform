"""Uncertainty sampling on predicted class probabilities.

Each strategy turns probs into a score (higher = label first) and takes the
top k. Ties keep pool order, so a selection is reproducible. All of them need
only one inference pass over the pool.
"""

import numpy as np

from selection.base import LabeledStats, PoolData, Strategy, register


def _require_probs(pool: PoolData) -> np.ndarray:
    if pool.probs is None:
        raise ValueError("this strategy needs probs in the pool scores")
    return pool.probs


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def top_k(pool: PoolData, scores: np.ndarray, k: int) -> list[str]:
    order = np.argsort(-scores, kind="stable")[: min(k, len(pool))]
    return [pool.ids[i] for i in order]


@register("least_confidence")
class LeastConfidence(Strategy):
    description = "Lowest top-1 probability first (1 - max p)."
    requires = frozenset({"probs"})

    def select(self, pool: PoolData, labeled: LabeledStats, k: int) -> list[str]:
        probs = _require_probs(pool)
        return top_k(pool, 1.0 - probs.max(axis=1), k)


@register("margin")
class Margin(Strategy):
    description = "Smallest gap between the top two class probabilities first."
    requires = frozenset({"probs"})

    def select(self, pool: PoolData, labeled: LabeledStats, k: int) -> list[str]:
        probs = _require_probs(pool)
        top2 = np.sort(probs, axis=1)[:, -2:]
        return top_k(pool, -(top2[:, 1] - top2[:, 0]), k)


@register("entropy")
class Entropy(Strategy):
    description = "Highest predictive entropy first."
    requires = frozenset({"probs"})

    def select(self, pool: PoolData, labeled: LabeledStats, k: int) -> list[str]:
        return top_k(pool, _entropy(_require_probs(pool)), k)


@register("class_balanced")
class ClassBalanced(Strategy):
    """Entropy weighted towards classes that are rare in the labeled set.

    freq(y)  = (n_y + 1) / (N + C)            Laplace-smoothed labeled class frequency
    score(x) = H(x) * sum_y p(y|x) / freq(y)
    """

    description = "Entropy x inverse labeled-class frequency: favours likely rare-class images."
    requires = frozenset({"probs"})

    def select(self, pool: PoolData, labeled: LabeledStats, k: int) -> list[str]:
        probs = _require_probs(pool)
        num_classes = probs.shape[1]
        counts = labeled.class_counts
        if counts.shape != (num_classes,):
            raise ValueError(f"class_counts has {counts.shape[0]} classes, probs has {num_classes}")
        freq = (counts + 1) / (counts.sum() + num_classes)
        return top_k(pool, _entropy(probs) * (probs / freq).sum(axis=1), k)
