import numpy as np
import pytest

from common.round import SelectionConfig
from selection import (
    STRATEGIES,
    LabeledStats,
    PoolData,
    Strategy,
    describe_strategies,
    from_config,
    get_strategy,
    preview_selection,
    register,
)

NONE = LabeledStats.empty(2)


def _pool(n: int) -> PoolData:
    return PoolData(ids=[f"img{i}" for i in range(n)])


def test_random_is_registered_and_needs_nothing():
    assert "random" in STRATEGIES
    assert STRATEGIES["random"].requires == frozenset()


def test_random_is_seeded_and_unique():
    pool = _pool(100)
    a = get_strategy("random", seed=7).select(pool, NONE, k=10)
    b = get_strategy("random", seed=7).select(pool, NONE, k=10)
    assert a == b
    assert len(set(a)) == 10
    assert set(a) <= set(pool.ids)


def test_random_k_larger_than_pool():
    assert len(get_strategy("random").select(_pool(3), NONE, k=10)) == 3


def test_unknown_strategy():
    with pytest.raises(KeyError, match="unknown strategy"):
        get_strategy("does-not-exist")


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):

        @register("random")
        class Dup(Strategy):
            def select(self, pool, labeled, k):
                return []


def test_pooldata_shape_check():
    with pytest.raises(ValueError):
        PoolData(ids=["a", "b"], probs=np.zeros((3, 4)))


def test_labeled_stats_from_sql_rows():
    stats = LabeledStats.from_counts({"rust": 3, "healthy": 7}, ["healthy", "rust", "miner"])
    assert stats.class_counts.tolist() == [7, 3, 0]
    assert stats.total == 10
    with pytest.raises(ValueError, match="not in the dataset label list"):
        LabeledStats.from_counts({"typo": 1}, ["healthy", "rust"])
    with pytest.raises(ValueError, match="non-negative"):
        LabeledStats(class_counts=np.array([1, -1]))


def test_strategy_params_are_validated():
    from pydantic import ValidationError

    from selection import StrategyParams

    class TopParams(StrategyParams):
        temperature: float = 1.0

    @register("_test_params")
    class WithParams(Strategy):
        Params = TopParams

        def select(self, pool, labeled, k):
            return pool.ids[:k]

    try:
        s = get_strategy("_test_params", params={"temperature": 2})
        assert s.params.temperature == 2.0
        with pytest.raises(ValidationError):
            get_strategy("_test_params", params={"temprature": 2})  # typo rejected
        with pytest.raises(ValidationError):
            get_strategy("random", params={"anything": 1})  # random takes no params
    finally:
        STRATEGIES.pop("_test_params")


def test_describe_strategies_lists_random():
    info = describe_strategies()["random"]
    assert info["requires"] == []
    assert info["params"] == {}
    assert info["description"]


def test_preview_summarises_selection():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.5, 0.5]])
    pool = PoolData(ids=["a", "b", "c", "d"], probs=probs)
    cfg = SelectionConfig(strategy="random", k=2, seed=1)
    p = preview_selection(cfg, pool, NONE, labels=["healthy", "rust"])
    assert len(p.selected) == 2
    assert sum(p.predicted_counts.values()) == 2
    assert p.pool_predicted_counts == {"healthy": 3, "rust": 1}
    assert p.pool_mean_confidence == pytest.approx(0.7)
    # Preview is a dry run: same config gives the same pick as the real round.
    assert p.selected == from_config(cfg).select(pool, NONE, 2)


def test_preview_rejects_missing_inputs():
    @register("_test_needs_emb")
    class NeedsEmb(Strategy):
        requires = frozenset({"embeddings"})

        def select(self, pool, labeled, k):
            return []

    try:
        with pytest.raises(ValueError, match="embeddings"):
            preview_selection(SelectionConfig(strategy="_test_needs_emb", k=1), _pool(3), NONE)
    finally:
        STRATEGIES.pop("_test_needs_emb")
