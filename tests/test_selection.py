import numpy as np
import pytest

from selection import STRATEGIES, PoolData, Strategy, get_strategy, register


def _pool(n: int) -> PoolData:
    return PoolData(ids=[f"img{i}" for i in range(n)])


def test_random_is_registered_and_needs_nothing():
    assert "random" in STRATEGIES
    assert STRATEGIES["random"].requires == frozenset()


def test_random_is_seeded_and_unique():
    pool = _pool(100)
    a = get_strategy("random", seed=7).select(pool, _pool(0), k=10)
    b = get_strategy("random", seed=7).select(pool, _pool(0), k=10)
    assert a == b
    assert len(set(a)) == 10
    assert set(a) <= set(pool.ids)


def test_random_k_larger_than_pool():
    assert len(get_strategy("random").select(_pool(3), _pool(0), k=10)) == 3


def test_unknown_strategy():
    with pytest.raises(KeyError, match="unknown strategy"):
        get_strategy("does-not-exist")


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):

        @register("random")
        class Dup(Strategy):
            def select(self, pool, labeled, k, **cfg):
                return []


def test_pooldata_shape_check():
    with pytest.raises(ValueError):
        PoolData(ids=["a", "b"], probs=np.zeros((3, 4)))
