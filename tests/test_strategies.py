import numpy as np
import pytest

from selection import STRATEGIES, PoolData, get_strategy

EMPTY = PoolData(ids=[])
PROBS = np.array(
    [
        [0.98, 0.01, 0.01],  # a: confident
        [0.40, 0.35, 0.25],  # b: low top-1, small margin, high entropy
        [0.50, 0.495, 0.005],  # c: smallest margin
        [0.34, 0.33, 0.33],  # d: most uncertain overall
        [0.70, 0.20, 0.10],  # e
    ]
)
POOL = PoolData(ids=list("abcde"), probs=PROBS)


def test_exactly_the_thesis_strategies_are_registered():
    assert set(STRATEGIES) == {"random", "least_confidence", "margin", "entropy", "class_balanced"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("least_confidence", ["d", "b"]),
        ("margin", ["c", "d"]),
        ("entropy", ["d", "b"]),
    ],
)
def test_uncertainty_ranking(name, expected):
    assert get_strategy(name).select(POOL, EMPTY, k=2) == expected


@pytest.mark.parametrize("name", ["least_confidence", "margin", "entropy", "class_balanced"])
def test_uncertainty_edge_cases(name):
    s = get_strategy(name)
    assert s.select(POOL, EMPTY, k=0) == []
    assert sorted(s.select(POOL, EMPTY, k=99)) == sorted(POOL.ids)
    assert "a" not in s.select(POOL, EMPTY, k=4)  # most confident image is picked last
    with pytest.raises(ValueError, match="probs"):
        s.select(PoolData(ids=["x"]), EMPTY, k=1)


def test_uncertainty_ties_keep_pool_order():
    pool = PoolData(ids=["x", "y", "z"], probs=np.full((3, 2), 0.5))
    assert get_strategy("entropy").select(pool, EMPTY, k=2) == ["x", "y"]


def _labeled(labels):
    return PoolData(ids=[f"L{i}" for i in range(len(labels))], labels=np.array(labels))


def test_class_balanced_matches_formula():
    labeled = _labeled([0, 0, 0, 0, 1])  # class 0: 4, class 1: 1, class 2: 0
    n = np.array([4, 1, 0])
    freq = (n + 1) / (n.sum() + 3)
    p = np.clip(PROBS, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=1)
    expected = h * (PROBS / freq).sum(axis=1)
    order = [POOL.ids[i] for i in np.argsort(-expected, kind="stable")]
    assert get_strategy("class_balanced").select(POOL, labeled, k=5) == order


def test_class_balanced_prefers_rare_class_at_equal_entropy():
    # Same entropy, different predicted class: the rare class should win.
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    pool = PoolData(ids=["common", "rare"], probs=probs)
    labeled = _labeled([0] * 20 + [1] * 5)
    assert get_strategy("entropy").select(pool, EMPTY, k=1) == ["common"]
    assert get_strategy("class_balanced").select(pool, labeled, k=1) == ["rare"]


def test_class_balanced_without_labels_is_uniform_entropy():
    # Nothing labeled: freq is uniform, so the order equals plain entropy.
    assert get_strategy("class_balanced").select(POOL, EMPTY, k=5) == get_strategy(
        "entropy"
    ).select(POOL, EMPTY, k=5)


def test_class_balanced_validates_labels():
    s = get_strategy("class_balanced")
    with pytest.raises(ValueError, match="labels"):
        s.select(POOL, PoolData(ids=["L0"]), k=1)
    with pytest.raises(ValueError, match="class indices"):
        s.select(POOL, _labeled([5]), k=1)
    with pytest.raises(ValueError, match="labels must have shape"):
        PoolData(ids=["a", "b"], labels=np.array([0]))
