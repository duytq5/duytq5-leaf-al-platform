"""Standalone trainings (design doc change 24): the start-training Lambda and
the al-train steps against a real Postgres (TEST_DATABASE_URL) and a fake S3,
in the order the state machine runs them."""

import json
import uuid

import pytest
from pydantic import ValidationError

from common.jobs import JobType, TrainJob, TrainResult, job_uuid
from common.s3keys import manifest_key, selection_key
from lambdas import control, training
from lambdas import round as steps
from tests.test_round_lambdas import _label, _run, _score_pool, seeded

COMMIT = "e" * 40
IMAGE = "ghcr.io/duytq5/duytq5-leaf-al-platform-worker:" + COMMIT
PATH = "configs/trainings/toy-fastvit.yaml"
CONFIG = {"dataset": "toy", "train": {"recipe": "timm_classifier", "params": {"epochs": 2}}}
RESULT = TrainResult(
    mlflow_name="toy",
    mlflow_version=4,
    mlflow_run_id="run4",
    model_uri="models:/toy/4",
    arch="fastvit_t8",
    metrics={"test_accuracy": 0.8},
)


@pytest.fixture
def env(pg_url, tmp_path):
    yield from seeded(pg_url, tmp_path)


def _event(config=None, path=PATH, commit=COMMIT) -> dict:
    return {
        "config": config or CONFIG,
        "config_path": path,
        "image": IMAGE,
        "started_by": "duytq5",
        "commit_sha": commit,
    }


def _start(conn, **kw) -> uuid.UUID:
    out = training.start_training(conn, training.StartTraining.model_validate(_event(**kw)))
    tid = uuid.UUID(out["training_id"])
    assert out["execution"] == {"training_id": str(tid), "image": IMAGE}
    return tid


def _status(conn, tid) -> str:
    return conn.execute("SELECT status FROM trainings WHERE id = %s", (tid,)).fetchone()[0]


def test_a_training_from_start_to_record(env):
    conn, store = env
    tid = _start(conn)
    assert tid == training.training_uuid(COMMIT, PATH)
    row = conn.execute(
        "SELECT t.status, t.started_by, t.commit_sha, v.version, t.train_config"
        " FROM trainings t JOIN dataset_versions v ON v.id = t.version_id WHERE t.id = %s",
        (tid,),
    ).fetchone()
    assert row[:4] == ("pending", "duytq5", COMMIT, 0)
    # Validated params with the recipe's defaults, as the worker will train.
    assert row[4]["recipe"] == "timm_classifier" and row[4]["params"]["epochs"] == 2
    assert "lr" in row[4]["params"]

    training.mark_running(conn, tid)
    training.mark_running(conn, tid)  # a retry is fine
    assert _status(conn, tid) == "running"

    out = training.snapshot(conn, store, tid)
    assert manifest_key("toy", 0) in store.objects
    # The state machine sends this as the train job; the worker must accept it.
    job = TrainJob(
        job_id=out["job_id"], image=IMAGE, task_token="t", training_id=tid, spec=out["spec"]
    )
    assert job.job_id == job_uuid(tid, JobType.TRAIN) and job.round_id is None
    assert job.spec.dataset == "toy" and job.spec.params["epochs"] == 2
    assert [c.code for c in job.spec.classes] == ["a", "b"]

    assert training.record(conn, tid, RESULT, IMAGE) == {"model": "models:/toy/4"}
    assert training.record(conn, tid, RESULT, IMAGE) == {"model": "models:/toy/4"}  # a retry
    assert conn.execute("SELECT image FROM model_versions").fetchone() == (IMAGE,)
    assert _status(conn, tid) == "succeeded"
    model = conn.execute(
        "SELECT mv.round_id, mv.training_id, d.name, v.version, mv.metrics, mv.labels"
        " FROM model_versions mv JOIN datasets d ON d.id = mv.dataset_id"
        " JOIN dataset_versions v ON v.id = mv.trained_on_version_id"
    ).fetchall()
    assert model == [
        (
            None,
            tid,
            "toy",
            0,
            {"test_accuracy": 0.8},
            [{"code": "a", "display_name": "Class A"}, {"code": "b", "display_name": "Class B"}],
        )
    ]


def test_starting_again_returns_the_same_training(env):
    conn, _ = env
    assert _start(conn) == _start(conn)
    assert conn.execute("SELECT count(*) FROM trainings").fetchone()[0] == 1


def test_a_training_does_not_block_a_round(env):
    """A training changes no data, so a round on the dataset may run alongside."""
    conn, _ = env
    _start(conn)
    out = control.start_round(
        conn,
        control.StartRound.model_validate(
            {
                **_event(),
                "config": {
                    "dataset": "toy",
                    "train": {"recipe": "timm_classifier"},
                    "selection": {"strategy": "random", "k": 3},
                },
                "config_path": "configs/rounds/r.yaml",
            }
        ),
    )
    assert out["round_id"]


def test_a_training_on_a_later_version_and_the_newest_model(env):
    """Rounds grow the dataset; the training trains on the newest version by
    default, and a later selection-only round scores with its model."""
    conn, store = env
    rid = uuid.uuid4()
    conn.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, strategy, k, seed,"
        " started_by, commit_sha) SELECT %s, dataset_id, id, 'random', 3, 7, 'o', %s"
        " FROM dataset_versions WHERE version = 0",
        (rid, COMMIT),
    )
    run = _run(rid).model_copy(update={"train": None, "model": "models:/toy/1"})
    steps.mark_running(conn, rid)
    steps.snapshot(conn, store, run)
    _score_pool(store, rid)
    steps.select(conn, store, rid)
    for i in json.loads(store.objects[selection_key(rid)])["image_ids"]:
        _label(conn, i, "a", rid)
    steps.merge(conn, store, run, None)

    tid = _start(conn)
    (version,) = conn.execute(
        "SELECT v.version FROM trainings t JOIN dataset_versions v ON v.id = t.version_id"
        " WHERE t.id = %s",
        (tid,),
    ).fetchone()
    assert version == 1
    training.mark_running(conn, tid)
    training.snapshot(conn, store, tid)
    training.record(conn, tid, RESULT, IMAGE)
    assert control._newest_model(conn, conn.execute("SELECT id FROM datasets").fetchone()[0]) == (
        "models:/toy/4"
    )


def test_fail_marks_the_training_failed(env):
    conn, _ = env
    tid = _start(conn)
    training.fail(conn, tid)
    training.fail(conn, tid)
    assert _status(conn, tid) == "failed"
    with pytest.raises(training.TrainingError, match="failed"):
        training.mark_running(conn, tid)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({**CONFIG, "dataset": "pepper"}, "no dataset pepper"),
        ({**CONFIG, "version": 9}, "no version 9"),
        ({**CONFIG, "train": {"recipe": "nope"}}, "unknown recipe"),
        ({**CONFIG, "train": {"recipe": "timm_classifier", "params": {"epoch": 1}}}, "epoch"),
        ({"dataset": "toy"}, "train"),
        ({**CONFIG, "selection": {"strategy": "random", "k": 1}}, "selection"),
    ],
)
def test_a_bad_training_is_refused(env, config, message):
    conn, _ = env
    with pytest.raises(training.BadTraining, match=message):
        _start(conn, config=config)
    assert conn.execute("SELECT count(*) FROM trainings").fetchone()[0] == 0


def test_the_event_is_checked():
    with pytest.raises(ValidationError):
        training.StartTraining.model_validate({**_event(), "commit_sha": "main"})


def test_handlers(env, monkeypatch):
    conn, store = env

    class _Conn:
        def __enter__(self):
            return conn

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(training, "open_db", _Conn)
    monkeypatch.setattr(training, "_store", lambda: store)
    out = training.start_training_handler(_event(), None)
    state = dict(out["execution"])
    state["training"] = training.mark_running_handler(state, None)
    state["snapshot"] = training.snapshot_handler(state, None)
    state["train_result"] = RESULT.model_dump(mode="json")
    assert training.record_handler(state, None) == {"model": "models:/toy/4"}
    assert conn.execute("SELECT image FROM model_versions").fetchone() == (IMAGE,)
    training.fail_handler(state, None)  # a finished training stays succeeded
    assert _status(conn, uuid.UUID(out["training_id"])) == "succeeded"
