"""Starting and cancelling rounds from GitHub (design doc F3, "Control through
GitHub"). The start-round workflow invokes lambdas.control.start_round, then
starts the al-round execution named by the returned round id; cancel-round
stops the execution, then invokes cancel_round.

Against a real Postgres (TEST_DATABASE_URL) with the toy dataset seeded.
"""

import json
import uuid

import pytest
from pydantic import ValidationError

from common.round import RoundExecution
from common.s3keys import selection_key
from lambdas import control
from lambdas import round as steps
from tests.test_round_lambdas import _image_status, _score_pool, _status, seeded

COMMIT = "d" * 40
IMAGE = "ghcr.io/duytq5/duytq5-leaf-al-platform-worker:" + COMMIT
PATH = "configs/rounds/toy-margin.yaml"
CONFIG = {
    "dataset": "toy",
    "train": {"recipe": "timm_classifier", "params": {"epochs": 2}},
    "selection": {"strategy": "margin", "params": {}, "k": 3, "seed": 7},
}


@pytest.fixture
def env(pg_url, tmp_path):
    yield from seeded(pg_url, tmp_path)


def _event(config=None, path=PATH, commit=COMMIT) -> dict:
    return {
        "config": config or CONFIG,
        "config_path": path,
        "image": IMAGE.replace(COMMIT, commit),
        "started_by": "duytq5",
        "commit_sha": commit,
    }


def _start(conn, **kw) -> dict:
    return control.start_round(conn, control.StartRound.model_validate(_event(**kw)))


def test_start_inserts_a_pending_round_and_returns_the_execution(env):
    conn, _ = env
    out = _start(conn)
    round_id = uuid.UUID(out["round_id"])
    assert round_id == control.round_uuid(COMMIT, PATH)
    row = conn.execute(
        "SELECT r.status, r.strategy, r.strategy_cfg, r.k, r.seed, r.started_by, r.commit_sha,"
        " v.version FROM rounds r JOIN dataset_versions v ON v.id = r.base_version_id"
        " WHERE r.id = %s",
        (round_id,),
    ).fetchone()
    assert row == ("pending", "margin", {}, 3, 7, "duytq5", COMMIT, 0)
    execution = RoundExecution.model_validate(out["execution"])
    assert (execution.round_id, execution.image) == (round_id, IMAGE)
    assert execution.train.recipe == "timm_classifier"
    # Validated params with the recipe's defaults, as the worker will train.
    assert execution.train.params["epochs"] == 2 and "lr" in execution.train.params


def test_starting_again_returns_the_same_round(env):
    """A re-run workflow: same commit and file, same round, nothing new."""
    conn, _ = env
    assert _start(conn) == _start(conn)
    assert conn.execute("SELECT count(*) FROM rounds").fetchone()[0] == 1


def test_one_active_round_per_dataset(env):
    conn, _ = env
    first = _start(conn)
    with pytest.raises(control.RoundConflict, match=first["round_id"]):
        _start(conn, path="configs/rounds/another.yaml")


def test_a_finished_round_frees_the_dataset(env):
    conn, store = env
    first = _start(conn)
    control.cancel_round(conn, store, uuid.UUID(first["round_id"]))
    assert _start(conn, path="configs/rounds/another.yaml")["round_id"] != first["round_id"]


def test_base_version_defaults_to_the_latest(env):
    conn, _ = env
    v0 = conn.execute("SELECT id FROM dataset_versions WHERE version = 0").fetchone()[0]
    rid = uuid.uuid4()  # a succeeded round made v1
    conn.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, strategy, k, seed, status,"
        " started_by, commit_sha) SELECT %s, dataset_id, id, 'random', 1, 1, 'succeeded',"
        " 'x', %s FROM dataset_versions WHERE id = %s",
        (rid, COMMIT, v0),
    )
    conn.execute(
        "INSERT INTO dataset_versions (dataset_id, version, parent_version_id,"
        " created_by_round_id) SELECT dataset_id, 1, id, %s FROM dataset_versions"
        " WHERE id = %s",
        (rid, v0),
    )
    out = _start(conn)
    (version,) = conn.execute(
        "SELECT v.version FROM rounds r JOIN dataset_versions v ON v.id = r.base_version_id"
        " WHERE r.id = %s",
        (uuid.UUID(out["round_id"]),),
    ).fetchone()
    assert version == 1


SELECTION_ONLY = {k: v for k, v in CONFIG.items() if k != "train"}


def _model(conn, version: int) -> None:
    """A model trained by an earlier (succeeded) round on toy."""
    rid = uuid.uuid4()
    conn.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, strategy, k, seed, status,"
        " started_by, commit_sha) SELECT %s, dataset_id, id, 'random', 1, 1, 'succeeded',"
        " 'x', %s FROM dataset_versions WHERE version = 0",
        (rid, COMMIT),
    )
    conn.execute(
        "INSERT INTO model_versions (round_id, dataset_id, trained_on_version_id, mlflow_name,"
        " mlflow_version, mlflow_run_id, arch, train_config, labels)"
        " SELECT %s, dataset_id, id, 'toy', %s, 'run', 'fastvit_t8', '{}',"
        ' \'[{"code": "a"}, {"code": "b"}]\' FROM dataset_versions WHERE version = 0',
        (rid, version),
    )


def _score_model(conn, out) -> str | None:
    return conn.execute(
        "SELECT score_model FROM rounds WHERE id = %s", (uuid.UUID(out["round_id"]),)
    ).fetchone()[0]


def test_a_round_that_trains_scores_with_its_own_model(env):
    conn, _ = env
    out = _start(conn)
    assert out["execution"]["model"] is None
    assert _score_model(conn, out) is None


def test_a_selection_only_round_scores_with_the_given_model(env):
    conn, _ = env
    out = _start(conn, config={**SELECTION_ONLY, "model": "models:/toy@champion"})
    execution = RoundExecution.model_validate(out["execution"])
    assert execution.train is None and execution.model == "models:/toy@champion"
    assert _score_model(conn, out) == "models:/toy@champion"


def test_a_selection_only_round_defaults_to_the_newest_model(env):
    conn, _ = env
    _model(conn, 1)
    _model(conn, 2)
    out = _start(conn, config=SELECTION_ONLY)
    assert out["execution"]["model"] == "models:/toy/2"
    assert _score_model(conn, out) == "models:/toy/2"


def test_a_selection_only_round_needs_a_model(env):
    conn, _ = env
    with pytest.raises(control.BadRound, match="no model"):
        _start(conn, config=SELECTION_ONLY)
    assert conn.execute("SELECT count(*) FROM rounds").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({**CONFIG, "dataset": "pepper"}, "no dataset pepper"),
        ({**CONFIG, "base_version": 9}, "no version 9"),
        ({**CONFIG, "train": {"recipe": "nope"}}, "unknown recipe"),
        ({**CONFIG, "train": {"recipe": "timm_classifier", "params": {"epoch": 1}}}, "epoch"),
        ({**CONFIG, "selection": {**CONFIG["selection"], "strategy": "nope"}}, "unknown strategy"),
        ({**CONFIG, "selection": {**CONFIG["selection"], "params": {"x": 1}}}, "x"),
    ],
)
def test_a_bad_round_is_refused(env, config, message):
    conn, _ = env
    with pytest.raises(control.BadRound, match=message):
        _start(conn, config=config)
    assert conn.execute("SELECT count(*) FROM rounds").fetchone()[0] == 0


def test_the_event_is_checked():
    with pytest.raises(ValidationError):
        control.StartRound.model_validate({**_event(), "commit_sha": "main"})
    with pytest.raises(ValidationError):
        control.StartRound.model_validate({**_event(), "image": "ubuntu:latest"})


# cancel


def test_cancel_puts_the_selected_images_back(env):
    conn, store = env
    rid = uuid.UUID(_start(conn)["round_id"])
    run = RoundExecution.model_validate(_start(conn)["execution"])
    steps.mark_running(conn, rid)
    steps.snapshot(conn, store, run)
    _score_pool(store, rid)
    steps.select(conn, store, rid)
    chosen = json.loads(store.objects[selection_key(rid)])["image_ids"]
    assert control.cancel_round(conn, store, rid) == {"round_id": str(rid), "status": "cancelled"}
    assert _status(conn, rid) == "cancelled"
    assert set(_image_status(conn, chosen).values()) == {"unlabeled"}
    assert conn.execute(
        "SELECT finished_at IS NOT NULL, label_task_token FROM rounds WHERE id = %s", (rid,)
    ).fetchone() == (True, None)


def test_cancelling_a_finished_round_changes_nothing(env):
    conn, store = env
    rid = uuid.UUID(_start(conn)["round_id"])
    steps.fail(conn, store, rid)
    assert control.cancel_round(conn, store, rid)["status"] == "failed"


def test_cancelling_an_unknown_round(env):
    conn, store = env
    with pytest.raises(control.BadRound):
        control.cancel_round(conn, store, uuid.uuid4())


def test_handlers(env, monkeypatch):
    conn, store = env

    class _Conn:
        def __enter__(self):
            return conn

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(control, "open_db", _Conn)
    monkeypatch.setattr(control, "_store", lambda: store)
    out = control.start_round_handler(_event(), None)
    assert control.cancel_round_handler({"round_id": out["round_id"]}, None)["status"] == (
        "cancelled"
    )


def test_round_input_builds_the_start_event(tmp_path, monkeypatch, capsys):
    from cli.main import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs/rounds").mkdir(parents=True)
    (tmp_path / "configs/rounds/toy.yaml").write_text(json.dumps(CONFIG))
    args = ["round-input", "configs/rounds/toy.yaml", "--started-by", "duytq5"]
    assert main([*args, "--commit-sha", COMMIT, "--image", IMAGE]) == 0
    ev = control.StartRound.model_validate_json(capsys.readouterr().out)
    assert ev.config == CONFIG and ev.config_path == "configs/rounds/toy.yaml"
