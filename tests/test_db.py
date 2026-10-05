"""Schema, migrations and seeding against a real Postgres (TEST_DATABASE_URL)."""

import uuid
from pathlib import Path

import psycopg
import pytest

from common.dataset import DatasetConfig
from db import connect
from db.migrate import migrate, migration_files
from db.seed import SeedError, seed_dataset
from selection.base import LABELED_COUNTS_SQL
from tests.test_seed import make_folder

CFG = DatasetConfig(name="toy", labels=["a", "b"], split={"val": 0.2, "test": 0.2, "seed": 3})


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts = 0

    def exists(self, key: str) -> bool:
        return key in self.objects

    def put_jpeg(self, key: str, path: Path) -> None:
        self.puts += 1
        self.objects[key] = path.read_bytes()


@pytest.fixture
def conn(pg_url):
    with connect(pg_url) as c:
        migrate(c)
        yield c


def test_migrate_is_idempotent(pg_url):
    with connect(pg_url) as c:
        assert migrate(c) == [p.name for p in migration_files()]
        assert migrate(c) == []


def test_seed_end_to_end(conn, tmp_path):
    src = make_folder(tmp_path, {"a": 20, "b": 10})
    store = FakeStore()
    report = seed_dataset(conn, CFG, src, store)

    assert report.uploaded == 30 and report.new_rows == 30
    assert all(k.startswith("raw/toy/") and k.endswith(".jpg") for k in store.objects)
    assert {s: sum(c.values()) for s, c in report.split_counts.items()} == {
        "train": 18,
        "val": 6,
        "test": 6,
    }
    rows = conn.execute(
        "SELECT split, status, source, count(*) FROM images GROUP BY 1, 2, 3 ORDER BY 1"
    ).fetchall()
    assert rows == [
        ("test", "labeled", "seed", 6),
        ("train", "unlabeled", "seed", 18),
        ("val", "labeled", "seed", 6),
    ]
    # Every image has ground truth for the oracle; only val/test have labels.
    assert conn.execute("SELECT count(*) FROM oracle_labels").fetchone() == (30,)
    assert conn.execute(
        "SELECT count(*) FROM labels l JOIN images i ON i.id = l.image_id WHERE i.split = 'train'"
    ).fetchone() == (0,)
    assert conn.execute("SELECT count(*) FROM labels WHERE round_id IS NULL").fetchone() == (12,)
    # Train labels start empty, so selection sees no labeled images yet.
    assert conn.execute(LABELED_COUNTS_SQL, {"dataset": "toy"}).fetchall() == []
    assert conn.execute("SELECT labels FROM datasets").fetchone() == (["a", "b"],)


def test_seed_rerun_is_a_noop(conn, tmp_path):
    src = make_folder(tmp_path, {"a": 20, "b": 10})
    store = FakeStore()
    seed_dataset(conn, CFG, src, store)
    before = conn.execute("SELECT id, split, status FROM images ORDER BY id").fetchall()

    report = seed_dataset(conn, CFG, src, store)
    assert report.uploaded == 0 and report.already_in_s3 == 30
    assert report.new_rows == 0 and report.existing_rows == 30
    assert store.puts == 30
    assert conn.execute("SELECT id, split, status FROM images ORDER BY id").fetchall() == before
    assert conn.execute("SELECT count(*) FROM labels").fetchone() == (12,)


def test_seed_keeps_first_split_when_images_are_added(conn, tmp_path):
    src = make_folder(tmp_path, {"a": 20, "b": 10})
    seed_dataset(conn, CFG, src, FakeStore())
    first = dict(conn.execute("SELECT sha256, split FROM images").fetchall())

    make_folder(tmp_path, {"a": 7}, start=20)
    seed_dataset(conn, CFG, src, FakeStore())
    after = dict(conn.execute("SELECT sha256, split FROM images").fetchall())
    assert len(after) == 37
    assert {s: after[s] for s in first} == first


def test_seed_refuses_a_changed_label_list(conn, tmp_path):
    seed_dataset(conn, CFG, make_folder(tmp_path, {"a": 5, "b": 5}), FakeStore())
    reordered = CFG.model_copy(update={"labels": ["b", "a"]})
    with pytest.raises(SeedError, match="cannot change"):
        seed_dataset(conn, reordered, tmp_path, FakeStore())


def test_seed_refuses_a_changed_ground_truth(conn, tmp_path):
    seed_dataset(conn, CFG, make_folder(tmp_path / "v1", {"a": 5, "b": 5}), FakeStore())
    v2 = make_folder(tmp_path / "v2", {"a": 4}, start=1)
    make_folder(v2, {"b": 5})
    (tmp_path / "v1" / "a" / "0.jpg").rename(v2 / "b" / "moved.jpg")
    with pytest.raises(SeedError, match="different label"):
        seed_dataset(conn, CFG, v2, FakeStore())


def _image(conn, split: str, status: str, sha: str = "0" * 64) -> None:
    conn.execute(
        "INSERT INTO datasets (name, labels) VALUES ('toy', '{a,b}') ON CONFLICT DO NOTHING"
    )
    conn.execute(
        "INSERT INTO images (dataset, sha256, s3_key, status, split, source)"
        " VALUES ('toy', %s, 'k', %s, %s, 'seed')",
        (sha, status, split),
    )


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("status", ["unlabeled", "queued", "pending"])
def test_eval_images_can_never_enter_the_pool(conn, split, status):
    with pytest.raises(psycopg.errors.CheckViolation):
        _image(conn, split, status)
    _image(conn, split, "labeled")
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE images SET status = %s", (status,))


def _round(conn, status: str = "pending") -> uuid.UUID:
    rid = uuid.uuid4()
    conn.execute(
        "INSERT INTO rounds (id, dataset, mode, strategy, k, seed, status, started_by)"
        " VALUES (%s, 'toy', 'simulation', 'random', 10, 42, %s, 'owner')",
        (rid, status),
    )
    return rid


def test_one_active_round_per_dataset(conn):
    _image(conn, "train", "unlabeled")
    first = _round(conn)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _round(conn)
    conn.execute("UPDATE rounds SET status = 'succeeded' WHERE id = %s", (first,))
    _round(conn, "running")


def test_label_insert_is_retry_safe(conn):
    _image(conn, "train", "unlabeled")
    (image_id,) = conn.execute("SELECT id FROM images").fetchone()
    rid = _round(conn)
    sql = (
        "INSERT INTO labels (image_id, label, round_id) VALUES (%s, 'a', %s)"
        " ON CONFLICT (image_id, round_id) DO NOTHING"
    )
    for _ in range(2):
        conn.execute(sql, (image_id, rid))
        conn.execute(sql, (image_id, None))
    assert conn.execute("SELECT count(*) FROM labels").fetchone() == (2,)
    assert conn.execute(LABELED_COUNTS_SQL, {"dataset": "toy"}).fetchall() == [("a", 1)]
