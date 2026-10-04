"""Core-set selection: k-center greedy on embeddings (Sener & Savarese, 2018).

Repeatedly picks the pool image farthest from everything already labeled or
selected, so the batch covers the embedding space instead of one region.
"""

from typing import Literal

import numpy as np

from selection.base import PoolData, Strategy, StrategyParams, register


class CoresetParams(StrategyParams):
    metric: Literal["euclidean", "cosine"] = "euclidean"


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _min_dist(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Distance from each point to its nearest center, in center chunks to bound memory."""
    out = np.full(len(points), np.inf)
    sq = (points**2).sum(axis=1)
    for start in range(0, len(centers), 1024):
        c = centers[start : start + 1024]
        d2 = sq[:, None] - 2.0 * points @ c.T + (c**2).sum(axis=1)[None, :]
        out = np.minimum(out, np.sqrt(np.maximum(d2, 0.0)).min(axis=1))
    return out


@register("coreset")
class Coreset(Strategy):
    description = "k-center greedy on embeddings: covers the pool's feature space."
    requires = frozenset({"embeddings"})
    Params = CoresetParams

    def select(self, pool: PoolData, labeled: PoolData, k: int) -> list[str]:
        if pool.embeddings is None:
            raise ValueError("coreset needs embeddings in the pool scores")
        if len(labeled) and labeled.embeddings is None:
            raise ValueError("coreset needs embeddings for the labeled set")
        k = min(k, len(pool))
        if k == 0:
            return []

        emb = pool.embeddings.astype(np.float64)
        lab = None if not len(labeled) else labeled.embeddings.astype(np.float64)
        if self.params.metric == "cosine":
            # Euclidean distance on unit vectors is monotonic in cosine distance.
            emb = _normalise(emb)
            lab = None if lab is None else _normalise(lab)

        if lab is None:
            first = int(self.rng.integers(len(pool)))
            dist = _min_dist(emb, emb[first : first + 1])
            dist[first] = -1.0
            chosen = [first]
        else:
            dist = _min_dist(emb, lab)
            chosen = []

        while len(chosen) < k:
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, _min_dist(emb, emb[nxt : nxt + 1]))
            dist[nxt] = -1.0  # never pick twice, even with duplicate embeddings
        return [pool.ids[i] for i in chosen]
