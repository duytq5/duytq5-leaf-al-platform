"""The label step and the webhook's database side (F2) against a real Postgres
(TEST_DATABASE_URL), a fake S3 and a fake Label Studio."""

import contextlib
import json
import uuid

import pytest

from common.s3keys import label_studio_export_key, raw_key
from lambdas import label_studio as ls
from lambdas import labeling
from lambdas import round as steps
from tests.test_round_lambdas import (
    RESULT,
    _image_status,
    _round,
    _run,
    _status,
    _to_selection,
    seeded,
)

BUCKET = "data"
HOOK = "https://api.example.com/v1/hooks/label-studio"
SECRET = "s" * 40


@pytest.fixture
def env(pg_url, tmp_path):
    yield from seeded(pg_url, tmp_path)


class FakeLabelStudio:
    def __init__(self) -> None:
        self.projects_: dict[int, dict] = {}
        self.storages: list[dict] = []
        self.hooks: list[dict] = []
        self.tasks: dict[int, list[dict]] = {}
        self.users: dict[int, dict] = {3: {"id": 3, "email": "expert@example.com"}}
        self.export_error: Exception | None = None

    def projects(self) -> list[dict]:
        return list(self.projects_.values())

    def project(self, project_id: int) -> dict:
        if project_id not in self.projects_:
            raise ls.LabelStudioNotFound(f"GET /api/projects/{project_id}/: HTTP 404")
        return {**self.projects_[project_id], "task_number": len(self.tasks[project_id])}

    def create_project(self, title: str, description: str, label_config: str) -> int:
        pid = len(self.projects_) + 1
        self.projects_[pid] = {
            "id": pid, "title": title, "description": description, "label_config": label_config,
        }  # fmt: skip
        self.tasks[pid] = []
        return pid

    def update_project(self, project_id: int, **fields) -> None:
        self.projects_[project_id].update(fields)

    def s3_storages(self, project_id: int) -> list[dict]:
        return [s for s in self.storages if s["project"] == project_id]

    def add_s3_storage(self, project_id: int, bucket: str, prefix: str) -> None:
        self.storages.append({"project": project_id, "bucket": bucket, "prefix": prefix})

    def webhooks(self, project_id: int) -> list[dict]:
        return [h for h in self.hooks if h["project"] == project_id]

    def add_webhook(self, project_id: int, url: str, headers: dict[str, str]) -> None:
        self.hooks.append({"project": project_id, "url": url, "headers": headers})

    def import_tasks(self, project_id: int, tasks: list[dict]) -> None:
        self.tasks[project_id].extend(tasks)

    def user(self, user_id: int) -> dict:
        if user_id not in self.users:
            raise ls.LabelStudioNotFound(f"GET /api/users/{user_id}/: HTTP 404")
        return self.users[user_id]

    def export(self, project_id: int) -> list[dict]:
        if self.export_error is not None:
            raise self.export_error
        if project_id not in self.projects_:
            raise ls.LabelStudioNotFound(f"GET /api/projects/{project_id}/export: HTTP 404")
        return [{**t, "annotations": [{"result": []}]} for t in self.tasks[project_id]]

    def delete_project(self, project_id: int) -> bool:
        return self.projects_.pop(project_id, None) is not None


def _start(conn, store, studio, rid, token="token-1") -> dict:
    return labeling.start_labeling(
        conn, store, studio, rid, token, bucket=BUCKET, webhook_url=HOOK, webhook_secret=SECRET
    )


def _labeling(conn, store):
    """A round that is waiting for its labels; returns it, its images and Label Studio."""
    rid = _round(conn)
    chosen = _to_selection(conn, store, rid)
    studio = FakeLabelStudio()
    _start(conn, store, studio, rid)
    return rid, chosen, studio


def _event(rid, image_id, choice="a", who="label-studio:1") -> ls.LabelEvent:
    return ls.LabelEvent(round_id=rid, image_id=image_id, choice=choice, labeled_by=who)


def test_start_labeling_makes_the_round_project(env):
    conn, store = env
    rid = _round(conn)
    chosen = _to_selection(conn, store, rid)
    studio = FakeLabelStudio()

    out = _start(conn, store, studio, rid)

    (project,) = studio.projects()
    assert out == {"project_id": project["id"], "tasks": 3}
    assert project["title"] == ls.project_title("toy", rid)
    assert project["description"] == ls.project_description(rid)
    assert 'alias="a"' in project["label_config"]
    assert 'alias="b"' in project["label_config"]
    assert conn.execute(
        "SELECT label_task_token, ls_project_id FROM rounds WHERE id = %s", (rid,)
    ).fetchone() == ("token-1", project["id"])
    # Label Studio signs image links itself, so they never expire in a task.
    assert studio.storages == [{"project": project["id"], "bucket": BUCKET, "prefix": "raw/"}]
    assert studio.hooks == [
        {"project": project["id"], "url": HOOK, "headers": {ls.SECRET_HEADER: SECRET}}
    ]

    tasks = studio.tasks[project["id"]]
    assert [t["data"]["image_id"] for t in tasks] == sorted(chosen)
    shas = dict(conn.execute("SELECT id, sha256 FROM images WHERE id = ANY(%s)", (chosen,)))
    for task in tasks:
        image_id = task["data"]["image_id"]
        assert task["data"]["round_id"] == str(rid)
        assert task["data"]["image"] == f"s3://{BUCKET}/{raw_key('toy', shas[image_id])}"
        # _score_pool: probs [(i % 7) / 7, 1 - (i % 7) / 7] for classes a, b.
        p_a = (image_id % 7) / 7
        (pred,) = task["predictions"]
        assert pred["result"][0]["value"]["choices"] == ["a" if p_a > 1 - p_a else "b"]
        assert pred["score"] == round(max(p_a, 1 - p_a), 4)


def test_start_labeling_again_reuses_everything(env):
    conn, store = env
    rid, chosen, studio = _labeling(conn, store)
    out = _start(conn, store, studio, rid, token="token-2")
    assert out["tasks"] == 3
    assert len(studio.projects()) == len(studio.storages) == len(studio.hooks) == 1
    assert len(studio.tasks[out["project_id"]]) == len(chosen)
    # The retried step has a new token; the webhook must resume that one.
    assert conn.execute("SELECT label_task_token FROM rounds WHERE id = %s", (rid,)).fetchone() == (
        "token-2",
    )


def test_start_labeling_finds_a_project_made_before_a_crash(env):
    conn, store = env
    rid = _round(conn)
    _to_selection(conn, store, rid)
    studio = FakeLabelStudio()
    studio.create_project("old title", ls.project_description(rid), "<View/>")
    out = _start(conn, store, studio, rid)
    assert out["project_id"] == 1
    assert len(studio.projects()) == 1


def test_start_labeling_needs_a_running_round(env):
    conn, store = env
    rid = _round(conn)
    with pytest.raises(steps.RoundError, match="pending"):
        _start(conn, store, FakeLabelStudio(), rid)


def test_only_train_images_reach_label_studio(env):
    conn, store = env
    rid = _round(conn)
    chosen = _to_selection(conn, store, rid)
    test_image = conn.execute("SELECT id FROM images WHERE split = 'test' LIMIT 1").fetchone()[0]
    # A selection file that names a test image (it never should).
    store.put(
        f"selections/{rid}.json",
        json.dumps({"round_id": str(rid), "image_ids": [*chosen, test_image]}).encode(),
    )
    studio = FakeLabelStudio()
    out = _start(conn, store, studio, rid)
    assert out["tasks"] == 3
    assert test_image not in [t["data"]["image_id"] for t in studio.tasks[out["project_id"]]]


def test_record_label_stores_each_label_once_and_resumes_when_done(env):
    conn, store = env
    rid, chosen, _ = _labeling(conn, store)

    assert labeling.record_label(conn, store, _event(rid, chosen[0], "a")) == {
        "status": "stored",
        "done": 1,
        "k": 3,
    }
    # Label Studio may send the same webhook twice.
    assert labeling.record_label(conn, store, _event(rid, chosen[0], "b"))["status"] == "stored"
    assert labeling.record_label(conn, store, _event(rid, chosen[1], "b"))["done"] == 2
    out = labeling.record_label(conn, store, _event(rid, chosen[2], ls.NOT_LEAF))
    assert out == {
        "status": "rejected",
        "done": 3,
        "k": 3,
        "task_token": "token-1",
        "project_id": 1,
        "title": ls.project_title("toy", rid, done=True),
    }

    labels = conn.execute(
        "SELECT image_id, c.code, source, labeled_by FROM labels l"
        " JOIN classes c ON c.id = l.class_id WHERE round_id = %s ORDER BY image_id",
        (rid,),
    ).fetchall()
    assert labels == sorted(
        [(chosen[0], "a", "expert", "label-studio:1"), (chosen[1], "b", "expert", "label-studio:1")]
    )
    assert _image_status(conn, chosen)[chosen[2]] == "rejected"

    # The merge step then builds the next version from these labels.
    assert steps.merge(conn, store, _run(rid), RESULT)["version"] == 1
    assert _status(conn, rid) == "succeeded"
    assert _image_status(conn, chosen) == {
        chosen[0]: "labeled",
        chosen[1]: "labeled",
        chosen[2]: "rejected",
    }
    # A late webhook for a finished round changes nothing.
    assert labeling.record_label(conn, store, _event(rid, chosen[0], "b")) == {"status": "ignored"}


def test_not_a_leaf_after_a_label_keeps_the_label(env):
    conn, store = env
    rid, chosen, _ = _labeling(conn, store)
    labeling.record_label(conn, store, _event(rid, chosen[0], "a"))
    assert labeling.record_label(conn, store, _event(rid, chosen[0], ls.NOT_LEAF))["status"] == (
        "ignored"
    )
    assert _image_status(conn, chosen)[chosen[0]] == "queued"
    labeling.record_label(conn, store, _event(rid, chosen[1], ls.NOT_LEAF))
    assert labeling.record_label(conn, store, _event(rid, chosen[1], "a"))["status"] == "ignored"
    assert conn.execute("SELECT count(*) FROM labels WHERE round_id = %s", (rid,)).fetchone() == (
        1,
    )


def test_record_label_refuses_unknown_codes(env):
    conn, store = env
    rid, chosen, _ = _labeling(conn, store)
    assert labeling.record_label(conn, store, _event(rid, chosen[0], "zzz")) == {
        "status": "unknown_code"
    }


def test_record_label_ignores_what_is_not_this_rounds_work(env):
    conn, store = env
    rid, chosen, _ = _labeling(conn, store)
    other = conn.execute(
        "SELECT id FROM images WHERE NOT (id = ANY(%s)) LIMIT 1", (chosen,)
    ).fetchone()[0]
    ignored = {"status": "ignored"}
    assert labeling.record_label(conn, store, _event(rid, other)) == ignored
    assert labeling.record_label(conn, store, _event(uuid.uuid4(), chosen[0])) == ignored
    # A round that has not reached the label step yet.
    conn.execute("UPDATE rounds SET status = 'failed' WHERE id = %s", (rid,))
    waiting = _round(conn)
    assert labeling.record_label(conn, store, _event(waiting, chosen[0])) == ignored
    assert conn.execute("SELECT count(*) FROM labels WHERE round_id IS NOT NULL").fetchone() == (0,)


# --- labeler email ---------------------------------------------------------------


def test_labeler_turns_a_label_studio_user_id_into_the_email():
    studio = FakeLabelStudio()
    assert labeling.labeler(studio, "label-studio:3") == "expert@example.com"


def test_labeler_keeps_what_it_cannot_improve():
    studio = FakeLabelStudio()
    # Already an email (the webhook expanded the user): no lookup.
    assert labeling.labeler(studio, "someone@example.com") == "someone@example.com"
    # The user is gone, or has no email: the id is better than nothing.
    assert labeling.labeler(studio, "label-studio:9") == "label-studio:9"
    studio.users = {4: {"id": 4, "email": ""}}
    assert labeling.labeler(studio, "label-studio:4") == "label-studio:4"
    assert labeling.labeler(studio, "label-studio:None") == "label-studio:None"


def test_labeler_keeps_the_id_when_label_studio_is_down():
    class Down(FakeLabelStudio):
        def user(self, user_id: int) -> dict:
            raise ls.LabelStudioError("GET /api/users/3/: Label Studio unreachable")

    assert labeling.labeler(Down(), "label-studio:3") == "label-studio:3"


def _handlers(monkeypatch, conn, store, studio) -> None:
    monkeypatch.setenv("DATA_BUCKET", BUCKET)
    monkeypatch.setattr(labeling, "open_db", lambda: contextlib.nullcontext(conn))
    monkeypatch.setattr(labeling, "S3Store", lambda bucket: store)
    monkeypatch.setattr(labeling, "_studio", lambda: studio)


def test_record_handler_stores_the_labelers_email(env, monkeypatch):
    conn, store = env
    rid, chosen, studio = _labeling(conn, store)
    _handlers(monkeypatch, conn, store, studio)
    event = {"round_id": str(rid), "image_id": chosen[0], "choice": "a",
             "labeled_by": "label-studio:3"}  # fmt: skip

    assert labeling.record_handler(event, None)["status"] == "stored"

    assert conn.execute("SELECT labeled_by FROM labels WHERE round_id = %s", (rid,)).fetchone() == (
        "expert@example.com",
    )


# --- closing the round's project ----------------------------------------------------


def _merged(conn, store):
    """A round that went through labels and merge; returns it, its images and
    Label Studio."""
    rid, chosen, studio = _labeling(conn, store)
    for image_id in chosen:
        labeling.record_label(conn, store, _event(rid, image_id, "a"))
    steps.merge(conn, store, _run(rid), RESULT)
    return rid, chosen, studio


def test_close_project_archives_the_project_then_deletes_it(env):
    conn, store = env
    rid, chosen, studio = _merged(conn, store)

    out = labeling.close_project(conn, store, studio, rid)

    key = label_studio_export_key(rid)
    assert key == f"labelstudio/{rid}.json"
    assert out == {"project_id": 1, "export": key, "tasks": 3, "deleted": True}
    exported = json.loads(store.get(key))
    assert sorted(t["data"]["image_id"] for t in exported) == sorted(chosen)
    assert all(t["annotations"] for t in exported)
    assert studio.projects() == []


def test_close_project_again_after_a_crash_is_fine(env):
    conn, store = env
    rid, _, studio = _merged(conn, store)
    labeling.close_project(conn, store, studio, rid)
    archived = store.get(label_studio_export_key(rid))

    # The project is gone: the export already in S3 is kept, not overwritten.
    out = labeling.close_project(conn, store, studio, rid)

    assert out == {"project_id": 1, "export": label_studio_export_key(rid), "tasks": None,
                   "deleted": False}  # fmt: skip
    assert store.get(label_studio_export_key(rid)) == archived


def test_close_project_keeps_the_project_when_the_export_fails(env):
    conn, store = env
    rid, _, studio = _merged(conn, store)
    studio.export_error = ls.LabelStudioError("GET /api/projects/1/export: HTTP 500")

    with pytest.raises(ls.LabelStudioError):
        labeling.close_project(conn, store, studio, rid)

    assert store.get(label_studio_export_key(rid)) is None
    assert [p["id"] for p in studio.projects()] == [1]


def test_close_project_keeps_the_project_when_the_export_misses_tasks(env):
    conn, store = env
    rid, _, studio = _merged(conn, store)

    class Short(FakeLabelStudio):
        def export(self, project_id: int) -> list[dict]:
            return super().export(project_id)[:2]

    short = Short()
    short.projects_, short.tasks = studio.projects_, studio.tasks

    with pytest.raises(ls.LabelStudioError, match="2 of 3 tasks"):
        labeling.close_project(conn, store, short, rid)
    assert store.get(label_studio_export_key(rid)) is None
    assert [p["id"] for p in short.projects()] == [1]


def test_close_project_fails_when_the_project_vanished_unarchived(env):
    conn, store = env
    rid, _, studio = _merged(conn, store)
    studio.projects_.clear()

    with pytest.raises(steps.RoundError, match="never archived"):
        labeling.close_project(conn, store, studio, rid)


def test_close_project_leaves_a_round_that_is_not_finished(env):
    conn, store = env
    rid, _, studio = _labeling(conn, store)

    with pytest.raises(steps.RoundError, match="running"):
        labeling.close_project(conn, store, studio, rid)
    assert [p["id"] for p in studio.projects()] == [1]
    assert store.get(label_studio_export_key(rid)) is None


def test_close_project_without_a_project_does_nothing(env):
    conn, store = env
    rid, _, studio = _merged(conn, store)
    conn.execute("UPDATE rounds SET ls_project_id = NULL WHERE id = %s", (rid,))

    out = labeling.close_project(conn, store, studio, rid)

    assert out == {"project_id": None, "export": None, "tasks": None, "deleted": False}
    assert [p["id"] for p in studio.projects()] == [1]


def test_close_handler_runs_close_project(env, monkeypatch):
    conn, store = env
    rid, _, studio = _merged(conn, store)
    _handlers(monkeypatch, conn, store, studio)

    assert labeling.close_handler({"round_id": str(rid)}, None)["deleted"] is True
    assert studio.projects() == []
