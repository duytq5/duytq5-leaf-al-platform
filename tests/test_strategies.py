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


def test_all_doc_strategies_registered():
    assert {"random", "least_confidence", "margin", "entropy", "coreset"} <= set(STRATEGIES)


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


@pytest.mark.parametrize("name", ["least_confidence", "margin", "entropy"])
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


def _clusters():
    # Three tight clusters; a good batch of 3 takes one from each.
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    emb = np.concatenate([c + 0.1 * rng.standard_normal((20, 2)) for c in centers])
    ids = [f"{c}{i}" for c in "ABC" for i in range(20)]
    return PoolData(ids=ids, embeddings=emb)


def test_coreset_covers_clusters():
    picked = get_strategy("coreset", seed=3).select(_clusters(), EMPTY, k=3)
    assert len(set(picked)) == 3
    assert {p[0] for p in picked} == {"A", "B", "C"}


def test_coreset_avoids_labeled_region():
    labeled = PoolData(ids=["L0"], embeddings=np.array([[0.0, 0.0]]))
    picked = get_strategy("coreset").select(_clusters(), labeled, k=2)
    assert {p[0] for p in picked} == {"B", "C"}


def test_coreset_seeded_and_unique_with_duplicates():
    pool = PoolData(ids=list("pqrs"), embeddings=np.zeros((4, 3)))
    a = get_strategy("coreset", seed=1).select(pool, EMPTY, k=4)
    assert sorted(a) == list("pqrs")
    assert a == get_strategy("coreset", seed=1).select(pool, EMPTY, k=4)


def test_coreset_cosine_metric():
    emb = np.array([[1.0, 0.0], [100.0, 0.0], [0.0, 1.0]])
    pool = PoolData(ids=["near", "far_same_dir", "other_dir"], embeddings=emb)
    labeled = PoolData(ids=["L"], embeddings=np.array([[1.0, 0.0]]))
    cos = get_strategy("coreset", params={"metric": "cosine"}).select(pool, labeled, k=1)
    euc = get_strategy("coreset").select(pool, labeled, k=1)
    assert cos == ["other_dir"]
    assert euc == ["far_same_dir"]


def test_coreset_needs_embeddings():
    with pytest.raises(ValueError, match="embeddings"):
        get_strategy("coreset").select(POOL, EMPTY, k=1)
    with pytest.raises(ValueError, match="labeled"):
        get_strategy("coreset").select(_clusters(), PoolData(ids=["L"]), k=1)
