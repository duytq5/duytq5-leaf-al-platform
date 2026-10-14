"""Model release and rollback (design doc F5).

A merged change to configs/releases/{dataset}.yaml starts the model-release
state machine: the prepare Lambda builds the export job, the worker exports
ONNX, and the record Lambda stores the export and a model_releases row.

Lambda steps against a real Postgres (TEST_DATABASE_URL).
"""

import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from cli.main import main
from common.jobs import ExportSpec
from common.release import ExportResult, ReleaseExecution, ReleaseFile, export_job_uuid
from lambdas import release
from tests.apidb import SHA, crop, dataset, model

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e" * 40
# tests.apidb.model() trains every version in a round of commit "c" * 40.
TRAINED_IMAGE = "ghcr.io/duytq5/duytq5-leaf-al-platform-worker:" + "c" * 40


def _execution(version: int, commit: str = COMMIT, *, model_name="durian-fastvit", ds="durian"):
    return ReleaseExecution(
        dataset=ds,
        model=model_name,
        version=version,
        released_by="duytq5",
        commit_sha=commit,
    )


def _export(version: int, sha: str = SHA) -> ExportResult:
    return ExportResult(
        onnx_s3_key=f"edge/durian-fastvit/{version}/model.onnx", onnx_sha256=sha, size_bytes=4242
    )


# The release file


def test_release_file_names_its_dataset(tmp_path):
    path = tmp_path / "durian.yaml"
    path.write_text("model: durian-fastvit\nversion: 3\n")
    name, release_file = ReleaseFile.from_yaml(path)
    assert (name, release_file) == ("durian", ReleaseFile(model="durian-fastvit", version=3))


@pytest.mark.parametrize(
    "text",
    [
        "model: durian-fastvit\nversion: 0\n",
        "model: durian/fastvit\nversion: 1\n",
        "model: durian-fastvit\n",
        "model: durian-fastvit\nversion: 1\nalias: champion\n",
    ],
)
def test_bad_release_files(tmp_path, text):
    path = tmp_path / "durian.yaml"
    path.write_text(text)
    with pytest.raises(ValidationError):
        ReleaseFile.from_yaml(path)


def test_execution_input_is_checked():
    with pytest.raises(ValidationError):
        _execution(1, commit="abc")
    with pytest.raises(ValidationError):
        ReleaseExecution(**{**_execution(1).model_dump(), "released_by": "-bad login"})


def test_export_job_id_is_stable_per_release():
    a = _execution(2)
    assert export_job_uuid(a) == export_job_uuid(_execution(2))
    # A rollback to the same version later is a new job: it moves the aliases again.
    assert export_job_uuid(a) != export_job_uuid(_execution(2, "f" * 40))


# al check-release (CI on every PR)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "configs/releases").mkdir(parents=True)
    (tmp_path / "configs/datasets").mkdir(parents=True)
    (tmp_path / "configs/datasets/durian.yaml").write_text("name: durian\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "configs/releases"


def test_check_release_ok(repo, capsys):
    (repo / "durian.yaml").write_text("model: durian-fastvit\nversion: 2\n")
    assert main(["check-release"]) == 0
    assert "ok: durian -> durian-fastvit/2" in capsys.readouterr().out


def test_check_release_rejects_an_unknown_dataset(repo, capsys):
    (repo / "pepper.yaml").write_text("model: pepper-fastvit\nversion: 1\n")
    assert main(["check-release"]) == 1
    assert "configs/datasets/pepper.yaml" in capsys.readouterr().out


def test_check_release_rejects_a_bad_file(repo, capsys):
    (repo / "durian.yaml").write_text("model: durian-fastvit\nversion: -1\n")
    assert main(["check-release"]) == 1
    assert "durian.yaml" in capsys.readouterr().out


def test_check_release_from_the_ops_repo(tmp_path, monkeypatch, capsys):
    """The ops repo's CI runs the pinned platform's `al` on its own release
    files, with the platform checkout's dataset configs."""
    (tmp_path / "ops/configs/releases").mkdir(parents=True)
    (tmp_path / "ops/configs/releases/durian.yaml").write_text("model: d\nversion: 1\n")
    (tmp_path / "platform/configs/datasets").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "ops")
    args = ["check-release", "--datasets", str(tmp_path / "platform/configs/datasets")]
    assert main(args) == 1
    assert "no dataset durian" in capsys.readouterr().out
    (tmp_path / "platform/configs/datasets/durian.yaml").write_text("name: durian\n")
    assert main(args) == 0


def test_check_release_with_no_files(repo, capsys):
    assert main(["check-release"]) == 0
    assert "no release files" in capsys.readouterr().out


# prepare


@pytest.fixture
def durian(db):
    ds = dataset(db, "durian", crop(db, "durian", "Sầu riêng"), "Lá sầu riêng")
    return {
        "dataset_id": ds,
        1: model(db, ds, "durian-fastvit", 1),
        2: model(db, ds, "durian-fastvit", 2, exported=False),
    }


def _releases(db) -> list[tuple]:
    return db.execute(
        "SELECT mv.mlflow_version, mr.action, mr.released_by, mr.commit_sha"
        " FROM model_releases mr JOIN model_versions mv ON mv.id = mr.model_version_id"
        " ORDER BY mr.id"
    ).fetchall()


def test_prepare_builds_the_export_job(db, durian):
    ev = _execution(2)
    out = release.prepare(db, ev)
    assert out["job_id"] == str(export_job_uuid(ev))
    # The export runs in the image that trained the model: the same recipe
    # code (preprocessing, transforms), whatever changed since.
    assert out["image"] == TRAINED_IMAGE
    spec = ExportSpec.model_validate(out["spec"])
    assert spec.model_uri == "models:/durian-fastvit/2"
    assert spec.output_prefix == "edge/durian-fastvit/2/"
    assert [c.code for c in spec.labels] == ["a", "b"]


@pytest.mark.parametrize(
    ("version", "model_name", "ds"),
    [
        (9, "durian-fastvit", "durian"),  # no such version
        (2, "other-model", "durian"),  # not this model
        (1, "durian-fastvit", "rocole"),  # no such dataset
    ],
)
def test_prepare_refuses_what_does_not_exist(db, durian, version, model_name, ds):
    with pytest.raises(release.ReleaseError):
        release.prepare(db, _execution(version, model_name=model_name, ds=ds))


def test_prepare_refuses_a_model_of_another_dataset(db, durian):
    fruit = dataset(db, "durian_fruit", crop(db, "fruit", "Quả"), "Quả sầu riêng")
    model(db, fruit, "fruit-fastvit", 1)
    with pytest.raises(release.ReleaseError):
        release.prepare(db, _execution(1, model_name="fruit-fastvit"))


# record


def test_first_release_is_a_promote_and_stores_the_export(db, durian):
    out = release.record(db, _execution(2), _export(2, "12" * 32))
    assert out == {"action": "promote", "model": "durian-fastvit", "version": 2}
    assert _releases(db) == [(2, "promote", "duytq5", COMMIT)]
    key, sha, size, exported = db.execute(
        "SELECT onnx_s3_key, onnx_sha256, size_bytes, exported_at FROM model_versions"
        " WHERE id = %s",
        (durian[2],),
    ).fetchone()
    assert (key, sha, size) == ("edge/durian-fastvit/2/model.onnx", "12" * 32, 4242)
    assert exported is not None


def test_releasing_an_earlier_version_again_is_a_rollback(db, durian):
    release.record(db, _execution(1, "1" * 40), _export(1))
    release.record(db, _execution(2, "2" * 40), _export(2))
    out = release.record(db, _execution(1, "3" * 40), _export(1))  # git revert
    assert out["action"] == "rollback"
    assert [r[:2] for r in _releases(db)] == [(1, "promote"), (2, "promote"), (1, "rollback")]


def test_record_runs_twice_without_a_second_row(db, durian):
    release.record(db, _execution(2), _export(2))
    assert release.record(db, _execution(2), _export(2))["action"] is None
    assert len(_releases(db)) == 1


def test_releasing_the_current_version_again_adds_nothing(db, durian):
    """A release file edited without changing the version (a comment, say)."""
    release.record(db, _execution(2), _export(2))
    assert release.record(db, _execution(2, "9" * 40), _export(2))["action"] is None
    assert len(_releases(db)) == 1


def test_an_export_with_other_bytes_is_refused(db, durian):
    """Version 1 was exported before (model() stored SHA); a different file
    would change what installed apps already verified."""
    with pytest.raises(release.ReleaseError):
        release.record(db, _execution(1), _export(1, "99" * 32))
    assert _releases(db) == []


def test_record_refuses_an_unknown_version(db, durian):
    with pytest.raises(release.ReleaseError):
        release.record(db, _execution(7), _export(7))


def test_the_released_model_is_what_the_app_gets(api, db, durian):
    from tests.apigw import call

    release.record(db, _execution(2), _export(2, "34" * 32))
    status, body, _ = call(api, "/v1/devices/config", query={"dataset": "durian"})
    assert status == 200
    assert (body["version"], body["sha256"]) == (2, "34" * 32)


# handlers


def test_handlers_validate_their_input(monkeypatch, db, durian):
    class _Conn:
        def __enter__(self):
            return db

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(release, "open_db", _Conn)
    ev = _execution(2).model_dump(mode="json")
    assert release.prepare_handler(ev, None)["job_id"] == str(export_job_uuid(_execution(2)))
    out = release.record_handler({**ev, "export": _export(2).model_dump()}, None)
    assert out["action"] == "promote"
    with pytest.raises(ValidationError):
        release.prepare_handler({**ev, "version": "x"}, None)


def test_job_ids_are_uuids():
    assert isinstance(export_job_uuid(_execution(1)), uuid.UUID)


def test_rerunning_an_old_release_commit_changes_nothing(db, durian):
    release.record(db, _execution(1, "1" * 40), _export(1))
    release.record(db, _execution(2, "2" * 40), _export(2))
    # The first release's workflow is re-run after the second one.
    assert release.record(db, _execution(1, "1" * 40), _export(1))["action"] is None
    assert [r[0] for r in _releases(db)] == [1, 2]


# al release-input (the release workflow builds the execution input with it)


def test_release_input_prints_the_execution_input(repo, capsys):
    import json

    (repo / "durian.yaml").write_text("model: durian-fastvit\nversion: 2\n")
    args = ["release-input", "configs/releases/durian.yaml", "--released-by", "duytq5"]
    assert main([*args, "--commit-sha", COMMIT]) == 0
    out = json.loads(capsys.readouterr().out)
    assert ReleaseExecution.model_validate(out) == _execution(2)


def test_release_input_refuses_a_bad_file(repo, capsys):
    (repo / "durian.yaml").write_text("model: durian-fastvit\nversion: 0\n")
    args = ["release-input", "configs/releases/durian.yaml", "--released-by", "duytq5"]
    assert main([*args, "--commit-sha", COMMIT]) == 1
