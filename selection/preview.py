"""Dry-run a selection so a human can compare strategies before committing a round.

Pure function over the latest score output: nothing is written, nothing is
queued for labeling.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from common.round import SelectionConfig
from selection.base import LabeledStats, PoolData, from_config


@dataclass
class SelectionPreview:
    strategy: str
    k: int
    selected: list[str]
    # Filled when the pool has probs: predicted-class counts and mean top-1 confidence.
    predicted_counts: dict[str, int] = field(default_factory=dict)
    pool_predicted_counts: dict[str, int] = field(default_factory=dict)
    mean_confidence: float | None = None
    pool_mean_confidence: float | None = None


def preview_selection(
    cfg: SelectionConfig,
    pool: PoolData,
    labeled: LabeledStats,
    labels: list[str] | None = None,
) -> SelectionPreview:
    strategy = from_config(cfg)
    missing = set(strategy.requires) - {
        name for name in ("probs", "embeddings") if getattr(pool, name) is not None
    }
    if missing:
        raise ValueError(f"strategy {cfg.strategy!r} needs {sorted(missing)} in the pool scores")

    selected = strategy.select(pool, labeled, cfg.k)
    out = SelectionPreview(strategy=cfg.strategy, k=cfg.k, selected=selected)
    if pool.probs is None:
        return out

    names = labels or [str(i) for i in range(pool.probs.shape[1])]
    top1 = pool.probs.argmax(axis=1)
    conf = pool.probs.max(axis=1)
    rows = np.array([pool.ids.index(i) for i in selected], dtype=int)

    out.pool_predicted_counts = dict(Counter(names[c] for c in top1))
    out.pool_mean_confidence = float(conf.mean()) if len(conf) else None
    out.predicted_counts = dict(Counter(names[c] for c in top1[rows]))
    out.mean_confidence = float(conf[rows].mean()) if len(rows) else None
    return out
