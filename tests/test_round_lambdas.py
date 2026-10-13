"""The al-round Lambda steps against a real Postgres (TEST_DATABASE_URL) and a
fake S3, run in the order the state machine runs them."""

import io
import json
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from common.dataset import CropsConfig, DatasetConfig
from common.jobs import JobType, TrainJob, TrainResult, job_uuid
from common.round import RoundExecution
from common.s3keys import manifest_key, pool_manifest_key, scores_key, selection_key
from db import connect
from db.migrate import migrate
from db.seed import seed_dataset
from lambdas import round as steps
from tests.test_seed import make_folder

COMMIT = "c" * 40
IMAGE = "ghcr.io/duytq5/duytq5-leaf-al-platform-worker:" + COMMIT
TRAIN = {
    "model": {"arch": "fastvit_t8"},
    "head": {"num_classes": 2},
    "data": {"dataset": "toy", "version": "v0"},
    "train": {"epochs": 1, "batch_size": 8, "lr": 3e-4},
}
RESULT = TrainResult(
    mlflow_name="toy",
    mlflow_version=1,
    mlflow_run_id="run1",
    model_uri="models:/toy/1",
    arch="fastvit_t8",
    metrics={"test_accuracy": 0.9},
)

CFG = DatasetConfig(
    name="toy",
    crop="plant",
    display_name="Toy leaves",
    classes=[{"code": "a", "display_name": "Class A"}, {"code": "b", "display_name": "Class B"}],
    split={"val": 0.2, "test": 0.2, "seed": 3, "initial_labeled": 0.3},
)
CROPS = CropsConfig(crops=[{"code": "plant", "display_name": "Plant"}])


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self.objects

    def put_jpeg(self, key, path) -> None:
        self.objects[key] = path.read_bytes()

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def put(self, key: str, body: bytes) -> None:
        self.objects[key] = body

    def uri(self, key: str) -> str:
        return f"s3://data/{key}"


def seeded(pg_url, tmp_path):
    """A migrated database with the toy dataset seeded, and the fake S3."""
    with connect(pg_url) as conn:
        migrate(conn)
        store = FakeStore()
        seed_dataset(conn, CFG, make_folder(tmp_path, {"a": 20, "b": 10}), store, CROPS)
        yield conn, store


@pytest.fixture
def env(pg_url, tmp_path):
    yield from seeded(pg_url, tmp_path)


def _round(conn, strategy: str = "random", k: int = 3) -> uuid.UUID:
    rid = uuid.uuid4()
    conn.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, mode, strategy, k, seed,"
        " started_by, commit_sha)"
        " SELECT %s, d.id, v.id, 'production', %s, %s, 7, 'owner', %s"
        " FROM datasets d JOIN dataset_versions v ON v.dataset_id = d.id AND v.version = 0"
        " WHERE d.name = 'toy'",
        (rid, strategy, k, COMMIT),
    )
    return rid


def _run(rid) -> RoundExecution:
    return RoundExecution(round_id=rid, image=IMAGE, train=TRAIN)


def _status(conn, rid) -> str:
    return conn.execute("SELECT status FROM rounds WHERE id = %s", (rid,)).fetchone()[0]


def _image_status(conn, ids) -> dict[int, str]:
    return dict(conn.execute("SELECT id, status FROM images WHERE id = ANY(%s)", (ids,)))


def _table(store, key) -> pa.Table:
    return pq.read_table(io.BytesIO(store.objects[key]))


def _score_pool(store, rid) -> list[int]:
    """What the worker's score job does: probs for every pool image."""
    ids = _table(store, pool_manifest_key(rid)).column("image_id").to_pylist()
    buf = io.BytesIO()
    probs = [[(i % 7) / 7, 1 - (i % 7) / 7] for i in ids]
    pq.write_table(pa.table({"image_id": ids, "probs": probs}), buf)
    store.put(scores_key(rid), buf.getvalue())
    return ids


def _label(conn, image_id: int, code: str, rid) -> None:
    conn.execute(
        "INSERT INTO labels (image_id, round_id, class_id, source, labeled_by)"
        " SELECT %s, %s, c.id, 'expert', 'expert1' FROM classes c"
        " JOIN datasets d ON d.id = c.dataset_id WHERE d.name = 'toy' AND c.code = %s",
        (image_id, rid, code),
    )


def _to_selection(conn, store, rid) -> list[int]:
    steps.mark_running(conn, rid)
    steps.snapshot(conn, store, _run(rid))
    _score_pool(store, rid)
    steps.select(conn, store, rid)
    return json.loads(store.objects[selection_key(rid)])["image_ids"]


def test_a_round_from_start_to_merge(env):
    conn, store = env
    rid = _round(conn)

    assert steps.mark_running(conn, rid) == {"dataset": "toy", "base_version": 0, "k": 3}
    assert _status(conn, rid) == "running"
    assert steps.mark_running(conn, rid)["k"] == 3  # a retry is fine

    out = steps.snapshot(conn, store, _run(rid))
    manifest = _table(store, manifest_key("toy", 0))
    v0 = conn.execute(
        "SELECT count(*) FROM dataset_version_labels vl JOIN dataset_versions v"
        " ON v.id = vl.version_id WHERE v.version = 0"
    ).fetchone()[0]
    assert manifest.num_rows == v0
    assert set(manifest.column("split").to_pylist()) == {"train", "val", "test"}
    assert all(k.startswith("raw/toy/") for k in manifest.column("key").to_pylist())
    pool = _table(store, pool_manifest_key(rid)).column("image_id").to_pylist()
    unlabeled = conn.execute(
        "SELECT count(*) FROM images WHERE split = 'train' AND status = 'unlabeled'"
    ).fetchone()[0]
    assert len(pool) == unlabeled == out["pool_size"]
    assert conn.execute("SELECT manifest_uri FROM dataset_versions").fetchone()[0] == (
        "s3://data/manifests/toy/v0.parquet"
    )
    # The state machine sends this as the train job; the worker must accept it.
    job = TrainJob(
        job_id=out["train"]["job_id"],
        image=IMAGE,
        task_token="t",
        round_id=rid,
        spec=out["train"]["spec"],
    )
    assert job.job_id == job_uuid(rid, JobType.TRAIN)
    assert [c.code for c in job.spec.classes] == ["a", "b"]
    assert out["score"]["outputs"] == ["probs"]
    assert out["score"]["output_uri"] == f"s3://data/{scores_key(rid)}"

    _score_pool(store, rid)
    assert steps.select(conn, store, rid) == {"selected": 3}
    chosen = json.loads(store.objects[selection_key(rid)])["image_ids"]
    assert set(_image_status(conn, chosen).values()) == {"queued"}
    assert steps.select(conn, store, rid) == {"selected": 3}  # a retry keeps the same set
    assert json.loads(store.objects[selection_key(rid)])["image_ids"] == chosen

    # The label step (tests/test_labeling.py) stores the token the webhook resumes.
    conn.execute("UPDATE rounds SET label_task_token = 'token-1' WHERE id = %s", (rid,))

    # The expert labels two images and marks the third "not a leaf".
    _label(conn, chosen[0], "a", rid)
    _label(conn, chosen[1], "b", rid)
    conn.execute("UPDATE images SET status = 'rejected' WHERE id = %s", (chosen[2],))

    assert steps.merge(conn, store, _run(rid), RESULT) == {"version": 1, "labels": v0 + 2}
    assert steps.merge(conn, store, _run(rid), RESULT) == {"version": 1, "labels": v0 + 2}
    assert _status(conn, rid) == "succeeded"
    assert _image_status(conn, chosen) == {
        chosen[0]: "labeled",
        chosen[1]: "labeled",
        chosen[2]: "rejected",
    }
    model = conn.execute(
        "SELECT mlflow_name, mlflow_version, arch, labels, metrics, train_config"
        " FROM model_versions WHERE round_id = %s",
        (rid,),
    ).fetchone()
    assert model[:3] == ("toy", 1, "fastvit_t8")
    assert model[3] == [
        {"code": "a", "display_name": "Class A"},
        {"code": "b", "display_name": "Class B"},
    ]
    assert model[4] == {"test_accuracy": 0.9}
    assert model[5] == TRAIN
    assert conn.execute(
        "SELECT label_task_token, finished_at IS NOT NULL FROM rounds WHERE id = %s", (rid,)
    ).fetchone() == (None, True)


def test_the_next_round_trains_on_the_merged_version(env):
    conn, store = env
    first = _round(conn)
    chosen = _to_selection(conn, store, first)
    for i in chosen:
        _label(conn, i, "a", first)
    steps.merge(conn, store, _run(first), RESULT)

    second = uuid.uuid4()
    conn.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, mode, strategy, k, seed,"
        " started_by, commit_sha)"
        " SELECT %s, dataset_id, id, 'production', 'random', 3, 7, 'owner', %s"
        " FROM dataset_versions WHERE version = 1",
        (second, COMMIT),
    )
    steps.mark_running(conn, second)
    steps.snapshot(conn, store, _run(second))
    labeled = set(_table(store, manifest_key("toy", 1)).column("image_id").to_pylist())
    assert set(chosen) <= labeled
    pool = set(_table(store, pool_manifest_key(second)).column("image_id").to_pylist())
    assert not pool & set(chosen)


def test_fail_puts_queued_images_back_and_is_repeatable(env):
    conn, store = env
    rid = _round(conn)
    chosen = _to_selection(conn, store, rid)

    steps.fail(conn, store, rid)
    steps.fail(conn, store, rid)
    assert _status(conn, rid) == "failed"
    assert set(_image_status(conn, chosen).values()) == {"unlabeled"}
    with pytest.raises(steps.RoundError, match="failed"):
        steps.mark_running(conn, rid)
    # A failed round frees the dataset for the next one.
    _round(conn)


def test_fail_before_selection(env):
    conn, store = env
    rid = _round(conn)
    steps.fail(conn, store, rid)
    assert _status(conn, rid) == "failed"


def test_steps_refuse_a_round_that_is_not_running(env):
    conn, store = env
    rid = _round(conn)
    with pytest.raises(steps.RoundError, match="pending"):
        steps.snapshot(conn, store, _run(rid))
    with pytest.raises(steps.RoundError, match="does not exist"):
        steps.mark_running(conn, uuid.uuid4())


def test_select_skips_images_that_left_the_pool(env):
    conn, store = env
    rid = _round(conn, k=100)
    steps.mark_running(conn, rid)
    steps.snapshot(conn, store, _run(rid))
    pool = _score_pool(store, rid)
    conn.execute("UPDATE images SET status = 'rejected' WHERE id = %s", (pool[0],))
    assert steps.select(conn, store, rid) == {"selected": len(pool) - 1}


def test_select_needs_the_scores(env):
    conn, store = env
    rid = _round(conn)
    steps.mark_running(conn, rid)
    with pytest.raises(steps.RoundError, match="no scores"):
        steps.select(conn, store, rid)


def test_uncertainty_strategy_asks_only_for_probs(env):
    conn, store = env
    rid = _round(conn, strategy="entropy")
    steps.mark_running(conn, rid)
    assert steps.snapshot(conn, store, _run(rid))["score"]["outputs"] == ["probs"]
    _score_pool(store, rid)
    assert steps.select(conn, store, rid) == {"selected": 3}


def test_lambda_connection_checks_the_certificate_name(monkeypatch, tmp_path):
    from psycopg.conninfo import conninfo_to_dict

    from lambdas import postgres

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PG_HOSTADDR", "10.0.1.23")
    monkeypatch.setenv("PG_PASSWORD", "secret")
    monkeypatch.setenv("PG_CA", "-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setattr(postgres, "CA_PATH", tmp_path / "ca.crt")
    postgres._conninfo.cache_clear()
    info = conninfo_to_dict(postgres._conninfo())
    postgres._conninfo.cache_clear()
    assert info["host"] == "leaf-al-postgres"  # what verify-full checks
    assert info["hostaddr"] == "10.0.1.23"  # where it connects
    assert info["sslmode"] == "verify-full"
    assert (tmp_path / "ca.crt").read_text().startswith("-----BEGIN CERTIFICATE")
