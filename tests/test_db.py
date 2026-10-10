"""Schema, migrations and seeding against a real Postgres (TEST_DATABASE_URL)."""

import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Jsonb

from common.dataset import DatasetConfig
from db import connect
from db.migrate import migrate, migration_files
from db.seed import SeedError, seed_dataset
from selection.base import LABELED_COUNTS_SQL
from tests.test_seed import make_folder

SNAPSHOT = [{"code": "a", "display_name": "Class A"}, {"code": "b", "display_name": "Class B"}]

# The 'a' class of the toy dataset, for raw SQL in tests.
CLASS_A = (
    "(SELECT c.id FROM classes c JOIN datasets d ON d.id = c.dataset_id"
    " WHERE d.name = 'toy' AND c.code = 'a')"
)

CFG = DatasetConfig(
    name="toy",
    classes=[{"code": "a", "display_name": "Class A"}, {"code": "b", "display_name": "Class B"}],
    split={"val": 0.2, "test": 0.2, "seed": 3},
)


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
    assert conn.execute(
        "SELECT source, count(*), count(labeled_by) FROM labels GROUP BY source"
    ).fetchall() == [("seed", 12, 0)]
    # v0 holds exactly the seed labels; it has no train labels, so selection
    # sees no labeled images yet.
    assert report.v0_added == 12
    v0 = _v0(conn)
    assert conn.execute(
        "SELECT count(*) FROM dataset_version_labels WHERE version_id = %s", (v0,)
    ).fetchone() == (12,)
    assert conn.execute(LABELED_COUNTS_SQL, {"version_id": v0}).fetchall() == []
    assert conn.execute(
        "SELECT code, position, display_name FROM classes ORDER BY position"
    ).fetchall() == [("a", 0, "Class A"), ("b", 1, "Class B")]


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
    reordered = CFG.model_copy(update={"classes": list(reversed(CFG.classes))})
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
    conn.execute("INSERT INTO datasets (name) VALUES ('toy') ON CONFLICT DO NOTHING")
    conn.execute(
        "INSERT INTO classes (dataset_id, code, position, display_name)"
        " SELECT d.id, c.code, c.pos, upper(c.code) FROM datasets d,"
        " (VALUES ('a', 0), ('b', 1)) AS c (code, pos) WHERE d.name = 'toy'"
        " ON CONFLICT DO NOTHING"
    )
    conn.execute(
        "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
        " SELECT id, %s, 'k', %s, %s, 'seed' FROM datasets WHERE name = 'toy'",
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


def _v0(conn) -> int:
    conn.execute(
        "INSERT INTO dataset_versions (dataset_id, version)"
        " SELECT id, 0 FROM datasets WHERE name = 'toy' ON CONFLICT DO NOTHING"
    )
    return conn.execute(
        "SELECT v.id FROM dataset_versions v JOIN datasets d ON d.id = v.dataset_id"
        " WHERE d.name = 'toy' AND v.version = 0"
    ).fetchone()[0]


def _round(conn, status: str = "pending", base: int | None = None) -> uuid.UUID:
    rid = uuid.uuid4()
    conn.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, mode, strategy, k, seed, status,"
        " started_by)"
        " SELECT %s, id, %s, 'simulation', 'random', 10, 42, %s, 'owner' FROM datasets"
        " WHERE name = 'toy'",
        (rid, base or _v0(conn), status),
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
        "INSERT INTO labels (image_id, class_id, round_id, source)"
        f" VALUES (%s, {CLASS_A}, %s, %s)"
        " ON CONFLICT (image_id, round_id) DO NOTHING"
    )
    for _ in range(2):
        conn.execute(sql, (image_id, rid, "oracle"))
        conn.execute(sql, (image_id, None, "seed"))
    assert conn.execute("SELECT count(*) FROM labels").fetchone() == (2,)


def test_every_table_has_an_id_primary_key(conn):
    rows = conn.execute(
        "SELECT tc.table_name, string_agg(kcu.column_name, ',')"
        " FROM information_schema.table_constraints tc"
        " JOIN information_schema.key_column_usage kcu"
        "   ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema"
        " WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema = 'public'"
        "   AND tc.table_name <> 'schema_migrations'"
        " GROUP BY tc.table_name"
    ).fetchall()
    assert dict(rows) == {
        t: "id"
        for t in (
            "datasets",
            "classes",
            "images",
            "rounds",
            "labels",
            "oracle_labels",
            "dataset_versions",
            "dataset_version_labels",
            "model_versions",
            "model_releases",
            "captures",
        )
    }


def test_natural_keys_stay_unique(conn):
    _image(conn, "train", "unlabeled")
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("INSERT INTO datasets (name) VALUES ('toy')")
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO classes (dataset_id, code, position, display_name)"
            " SELECT id, 'a', 5, 'dup' FROM datasets WHERE name = 'toy'"
        )
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO classes (dataset_id, code, position, display_name)"
            " SELECT id, 'c', 0, 'same position' FROM datasets WHERE name = 'toy'"
        )
    (image_id,) = conn.execute("SELECT id FROM images").fetchone()
    oracle = f"INSERT INTO oracle_labels (image_id, class_id) VALUES (%s, {CLASS_A})"
    conn.execute(oracle, (image_id,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(oracle, (image_id,))
    model = _model(conn, _round(conn))
    capture = (
        "INSERT INTO captures (capture_id, dataset_id, sha256, user_sub,"
        " model_version_id, top1_class_id, confidence, probs, captured_at)"
        f" SELECT %s, id, %s, 'user-sub', %s, {CLASS_A}, 0.6,"
        """ '{"a": 0.6, "b": 0.4}', now()"""
        " FROM datasets WHERE name = 'toy'"
        " ON CONFLICT (capture_id) DO NOTHING"
    )
    cid = uuid.uuid4()
    for _ in range(2):
        conn.execute(capture, (cid, "1" * 64, model))
    assert conn.execute("SELECT count(*) FROM captures").fetchone() == (1,)


@pytest.mark.parametrize(
    ("source", "with_round", "labeled_by", "ok"),
    [
        ("seed", False, None, True),
        ("oracle", True, None, True),
        ("expert", True, "labeler1", True),
        ("seed", True, None, False),  # seed labels belong to no round
        ("oracle", False, None, False),  # oracle and expert labels belong to a round
        ("expert", True, None, False),  # an expert label names its labeler
        ("oracle", True, "labeler1", False),  # only experts have a labeler
        ("human", True, None, False),
    ],
)
def test_label_source_rules(conn, source, with_round, labeled_by, ok):
    _image(conn, "train", "unlabeled")
    (image_id,) = conn.execute("SELECT id FROM images").fetchone()
    rid = _round(conn) if with_round else None
    insert = lambda: conn.execute(  # noqa: E731
        "INSERT INTO labels (image_id, class_id, round_id, source, labeled_by)"
        f" VALUES (%s, {CLASS_A}, %s, %s, %s)",
        (image_id, rid, source, labeled_by),
    )
    if ok:
        insert()
    else:
        with pytest.raises(psycopg.errors.CheckViolation):
            insert()


def _label(conn, image_id: int, code: str, round_id) -> int:
    return conn.execute(
        "INSERT INTO labels (image_id, class_id, round_id, source)"
        " SELECT %s, c.id, %s, 'oracle' FROM classes c JOIN datasets d ON d.id = c.dataset_id"
        " WHERE d.name = 'toy' AND c.code = %s RETURNING id",
        (image_id, round_id, code),
    ).fetchone()[0]


def _next_version(conn, parent: int, round_id) -> int:
    """What the merge step will do: copy the parent's rows, then add or replace
    rows with this round's labels."""
    with conn.transaction():
        (vid,) = conn.execute(
            "INSERT INTO dataset_versions (dataset_id, version, parent_version_id,"
            " created_by_round_id)"
            " SELECT dataset_id, version + 1, id, %s FROM dataset_versions WHERE id = %s"
            " RETURNING id",
            (round_id, parent),
        ).fetchone()
        conn.execute(
            "INSERT INTO dataset_version_labels (version_id, image_id, label_id)"
            " SELECT %s, image_id, label_id FROM dataset_version_labels WHERE version_id = %s",
            (vid, parent),
        )
        conn.execute(
            "INSERT INTO dataset_version_labels (version_id, image_id, label_id)"
            " SELECT %s, image_id, id FROM labels WHERE round_id = %s"
            " ON CONFLICT (version_id, image_id) DO UPDATE SET label_id = EXCLUDED.label_id",
            (vid, round_id),
        )
    return vid


def test_versions_keep_strategies_and_rollbacks_apart(conn, tmp_path):
    seed_dataset(conn, CFG, make_folder(tmp_path, {"a": 20, "b": 10}), FakeStore())
    v0 = _v0(conn)
    train = [r[0] for r in conn.execute("SELECT id FROM images WHERE split = 'train' ORDER BY id")]

    # Round 1 on v0 labels two images -> v1.
    r1 = _round(conn, base=v0)
    _label(conn, train[0], "a", r1)
    _label(conn, train[1], "b", r1)
    conn.execute("UPDATE rounds SET status = 'succeeded' WHERE id = %s", (r1,))
    v1 = _next_version(conn, v0, r1)
    assert sorted(conn.execute(LABELED_COUNTS_SQL, {"version_id": v1}).fetchall()) == [
        ("a", 1),
        ("b", 1),
    ]

    # Round 2 on v1 corrects image 0 (a new row, not an update) -> v2.
    r2 = _round(conn, base=v1)
    _label(conn, train[0], "b", r2)
    conn.execute("UPDATE rounds SET status = 'succeeded' WHERE id = %s", (r2,))
    v2 = _next_version(conn, v1, r2)
    assert conn.execute(LABELED_COUNTS_SQL, {"version_id": v2}).fetchall() == [("b", 2)]

    # Older versions are unchanged, and another strategy starting from v0 sees nothing.
    assert sorted(conn.execute(LABELED_COUNTS_SQL, {"version_id": v1}).fetchall()) == [
        ("a", 1),
        ("b", 1),
    ]
    assert conn.execute(LABELED_COUNTS_SQL, {"version_id": v0}).fetchall() == []
    assert conn.execute(
        "SELECT version, parent_version_id FROM dataset_versions WHERE id = %s", (v2,)
    ).fetchone() == (2, v1)
    # A round creates at most one version.
    with pytest.raises(psycopg.errors.UniqueViolation):
        _next_version(conn, v1, r2)


def test_labels_are_append_only(conn):
    _image(conn, "train", "unlabeled")
    (image_id,) = conn.execute("SELECT id FROM images").fetchone()
    _label(conn, image_id, "a", _round(conn))
    for sql in (
        "UPDATE labels SET labeled_by = 'x'",
        "DELETE FROM labels",
        "TRUNCATE labels CASCADE",
    ):
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            conn.execute(sql)


def test_version_label_must_belong_to_its_image(conn):
    _image(conn, "train", "unlabeled", sha="0" * 64)
    _image(conn, "train", "unlabeled", sha="1" * 64)
    first, second = (r[0] for r in conn.execute("SELECT id FROM images ORDER BY id"))
    label_id = _label(conn, first, "a", _round(conn))
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute(
            "INSERT INTO dataset_version_labels (version_id, image_id, label_id)"
            " VALUES (%s, %s, %s)",
            (_v0(conn), second, label_id),
        )


def test_only_v0_has_no_parent(conn):
    _image(conn, "train", "unlabeled")
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO dataset_versions (dataset_id, version)"
            " SELECT id, 1 FROM datasets WHERE name = 'toy'"
        )


def test_seed_refuses_to_change_v0_once_a_round_used_it(conn, tmp_path):
    seed_dataset(conn, CFG, make_folder(tmp_path, {"a": 20, "b": 10}), FakeStore())
    _round(conn)
    make_folder(tmp_path, {"a": 10}, start=20)
    with pytest.raises(SeedError, match="trained on v0"):
        seed_dataset(conn, CFG, tmp_path, FakeStore())


def _model(conn, round_id, mlflow_version: int = 1) -> int:
    return conn.execute(
        "INSERT INTO model_versions (round_id, trained_on_version_id, mlflow_name,"
        " mlflow_version, mlflow_run_id, arch, train_config, labels)"
        " SELECT id, base_version_id, 'leaf-disease', %s, 'run', 'fastvit_t8', '{}', %s"
        " FROM rounds WHERE id = %s RETURNING id",
        (mlflow_version, Jsonb(SNAPSHOT), round_id),
    ).fetchone()[0]


def test_one_model_per_round_and_export_columns_together(conn):
    _image(conn, "train", "unlabeled")
    rid = _round(conn)
    model = _model(conn, rid)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _model(conn, rid, mlflow_version=2)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE model_versions SET onnx_s3_key = 'k' WHERE id = %s", (model,))
    conn.execute(
        "UPDATE model_versions SET onnx_s3_key = 'edge/leaf-disease/1/model.onnx',"
        " onnx_sha256 = %s, size_bytes = 100, exported_at = now() WHERE id = %s",
        ("2" * 64, model),
    )


def test_latest_release_is_the_champion_and_releases_are_append_only(conn):
    _image(conn, "train", "unlabeled")
    first = _round(conn)
    conn.execute("UPDATE rounds SET status = 'succeeded' WHERE id = %s", (first,))
    m1 = _model(conn, first, 1)
    m2 = _model(conn, _round(conn), 2)
    release = (
        "INSERT INTO model_releases (model_version_id, action, released_by)"
        " VALUES (%s, %s, 'owner')"
    )
    conn.execute(release, (m1, "promote"))
    conn.execute(release, (m2, "promote"))
    conn.execute(release, (m1, "rollback"))
    champion = conn.execute(
        "SELECT r.model_version_id FROM model_releases r"
        " JOIN model_versions m ON m.id = r.model_version_id"
        " JOIN rounds ro ON ro.id = m.round_id"
        " JOIN datasets d ON d.id = ro.dataset_id"
        " WHERE d.name = 'toy' ORDER BY r.released_at DESC, r.id DESC LIMIT 1"
    ).fetchone()
    assert champion == (m1,)
    for sql in ("UPDATE model_releases SET action = 'promote'", "DELETE FROM model_releases"):
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            conn.execute(sql)


def test_model_class_snapshot_is_a_json_array(conn):
    _image(conn, "train", "unlabeled")
    model = _model(conn, _round(conn))
    assert conn.execute("SELECT labels FROM model_versions WHERE id = %s", (model,)).fetchone() == (
        SNAPSHOT,
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "UPDATE model_versions SET labels = %s WHERE id = %s",
            (Jsonb({"code": "a"}), model),
        )


def test_query_indexes_exist(conn):
    rows = conn.execute(
        "SELECT tablename, indexdef FROM pg_indexes WHERE schemaname = 'public'"
    ).fetchall()
    defs = {(t, d.split(" USING btree ")[1]) for t, d in rows}
    assert ("images", "(dataset_id, split, status)") in defs
    assert ("labels", "(round_id)") in defs
    assert ("model_releases", "(released_at DESC, id DESC)") in defs
    assert ("captures", "(user_sub, created_at)") in defs


def test_rejected_is_a_train_status_only(conn):
    _image(conn, "train", "rejected")
    with pytest.raises(psycopg.errors.CheckViolation):
        _image(conn, "test", "rejected", sha="1" * 64)
