"""Read-only API for admins and labelers (design doc F3, report 10.5 and 10.6):

  GET /v1/rounds/active                  labelers, admins
  GET /v1/rounds/{round_id}              admins
  GET /v1/datasets/{name}/versions       admins
  GET /v1/datasets/{name}/versions/{n}   admins (paginated)

The API Lambda has no route to Step Functions, so a round's phase comes from
Postgres and its selection file in S3 (selections/{round_id}.json): before
the selection exists the machine steps run (`training`), then the expert
labels (`awaiting_labels`), then the round merges (`merging`).
"""

import json
import uuid

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mangum")

from api.app import app
from api.deps import get_data_store
from tests.apidb import LABEL_STUDIO_URL, MLFLOW_URL, crop, dataset, model
from tests.apigw import call

ADMIN = {"sub": "a-1", "cognito:groups": "[admins]"}
LABELER = {"sub": "l-1", "cognito:groups": "[labelers]"}
UPLOADER = {"sub": "u-1", "cognito:groups": "[uploaders]"}
COMMIT = "d" * 40


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)


@pytest.fixture
def store(api):
    fake = FakeStore()
    app.dependency_overrides[get_data_store] = lambda: fake
    return fake


@pytest.fixture
def durian(db):
    return dataset(db, "durian", crop(db, "durian", "Sầu riêng"), "Lá sầu riêng")


def _image(db, dataset_id: int, n: int, split="train", status="unlabeled") -> int:
    return db.execute(
        "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
        " VALUES (%s, %s, %s, %s, %s, 'seed') RETURNING id",
        (dataset_id, f"{n:064x}", f"raw/durian/{n:064x}.jpg", status, split),
    ).fetchone()[0]


def _class(db, dataset_id: int, code: str) -> int:
    return db.execute(
        "SELECT id FROM classes WHERE dataset_id = %s AND code = %s", (dataset_id, code)
    ).fetchone()[0]


def _label(db, image_id: int, class_id: int, round_id=None) -> int:
    source, by = ("expert", "ls-7") if round_id else ("seed", None)
    return db.execute(
        "INSERT INTO labels (image_id, round_id, class_id, source, labeled_by)"
        " VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (image_id, round_id, class_id, source, by),
    ).fetchone()[0]


def _version(db, dataset_id: int, n: int) -> int:
    return db.execute(
        "SELECT id FROM dataset_versions WHERE dataset_id = %s AND version = %s",
        (dataset_id, n),
    ).fetchone()[0]


def _in_version(db, version_id: int, image_id: int, label_id: int) -> None:
    db.execute(
        "INSERT INTO dataset_version_labels (version_id, image_id, label_id) VALUES (%s, %s, %s)",
        (version_id, image_id, label_id),
    )


def _round(db, dataset_id: int, status="running", k=3, token=None, project=None) -> uuid.UUID:
    round_id = uuid.uuid4()
    db.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, mode, strategy, k, seed, status,"
        " started_by, commit_sha, label_task_token, ls_project_id)"
        " VALUES (%s, %s, %s, 'production', 'margin', %s, 7, %s, 'duytq5', %s, %s, %s)",
        (round_id, dataset_id, _version(db, dataset_id, 0), k, status, COMMIT, token, project),
    )
    return round_id


def _select(store, db, round_id, image_ids) -> None:
    store.objects[f"selections/{round_id}.json"] = json.dumps(
        {"round_id": str(round_id), "image_ids": image_ids}
    ).encode()
    db.execute("UPDATE images SET status = 'queued' WHERE id = ANY(%s)", (image_ids,))


# Who may read


@pytest.mark.parametrize(
    ("path", "claims", "status"),
    [
        ("/v1/rounds/active", None, 401),
        ("/v1/rounds/active", UPLOADER, 403),
        ("/v1/rounds/active", LABELER, 200),
        ("/v1/rounds/active", ADMIN, 200),
        (f"/v1/rounds/{uuid.uuid4()}", LABELER, 403),
        ("/v1/datasets/durian/versions", LABELER, 403),
        ("/v1/datasets/durian/versions/0", UPLOADER, 403),
        ("/v1/datasets/durian/versions", None, 401),
    ],
)
def test_groups(api, store, durian, path, claims, status):
    assert call(api, path, claims=claims)[0] == status


# GET /v1/rounds/active


def test_no_active_round(api, store, durian):
    assert call(api, "/v1/rounds/active", claims=LABELER)[:2] == (200, {"rounds": []})


def test_finished_rounds_are_not_active(api, store, db, durian):
    model(db, durian, "durian", 1)  # a succeeded round
    _round(db, durian, status="failed")
    assert call(api, "/v1/rounds/active", claims=LABELER)[1] == {"rounds": []}


def test_a_round_before_selection_is_training(api, store, db, durian):
    round_id = _round(db, durian)
    _, body, _ = call(api, "/v1/rounds/active", claims=LABELER)
    assert body == {
        "rounds": [
            {
                "round_id": str(round_id),
                "dataset": "durian",
                "phase": "training",
                "k": 3,
                "labeled": 0,
                "label_studio_url": None,
            }
        ]
    }


def test_a_pending_round_is_starting(api, store, db, durian):
    _round(db, durian, status="pending")
    assert call(api, "/v1/rounds/active", claims=ADMIN)[1]["rounds"][0]["phase"] == "starting"


def test_a_round_waiting_for_labels_shows_progress(api, store, db, durian):
    round_id = _round(db, durian, token="tok", project=7)
    ids = [_image(db, durian, n) for n in range(1, 4)]
    _select(store, db, round_id, ids)
    _label(db, ids[0], _class(db, durian, "a"), round_id)
    db.execute("UPDATE images SET status = 'rejected' WHERE id = %s", (ids[1],))  # not a leaf
    (active,) = call(api, "/v1/rounds/active", claims=LABELER)[1]["rounds"]
    assert active["phase"] == "awaiting_labels"
    assert active["labeled"] == 2
    assert active["label_studio_url"] == f"{LABEL_STUDIO_URL}/projects/7"


def test_a_fully_labeled_round_is_merging(api, store, db, durian):
    round_id = _round(db, durian, k=2, token="tok", project=7)
    ids = [_image(db, durian, n) for n in range(1, 3)]
    _select(store, db, round_id, ids)
    for image_id in ids:
        _label(db, image_id, _class(db, durian, "b"), round_id)
    (active,) = call(api, "/v1/rounds/active", claims=LABELER)[1]["rounds"]
    assert (active["phase"], active["labeled"]) == ("merging", 2)


def test_a_selection_smaller_than_k_counts_what_was_selected(api, store, db, durian):
    # select skips images that left the pool, so a round can have fewer than k.
    round_id = _round(db, durian, k=5, token="tok", project=7)
    (image_id,) = [_image(db, durian, 1)]
    _select(store, db, round_id, [image_id])
    _label(db, image_id, _class(db, durian, "a"), round_id)
    assert call(api, "/v1/rounds/active", claims=LABELER)[1]["rounds"][0]["phase"] == "merging"


# GET /v1/rounds/{round_id}


def test_round_status_of_a_running_round(api, store, db, durian):
    round_id = _round(db, durian, token="tok", project=7)
    status, body, _ = call(api, f"/v1/rounds/{round_id}", claims=ADMIN)
    assert status == 200
    created_at = body.pop("created_at")
    assert created_at.startswith("20")
    assert body == {
        "round_id": str(round_id),
        "dataset": "durian",
        "status": "running",
        "phase": "training",
        "strategy": "margin",
        "k": 3,
        "labeled": 0,
        "base_version": 0,
        "started_by": "duytq5",
        "commit_sha": COMMIT,
        "mlflow_run_url": None,
        "finished_at": None,
    }


def test_round_status_of_a_finished_round(api, store, db, durian):
    model(db, durian, "durian", 1)
    (round_id,) = db.execute("SELECT id FROM rounds").fetchone()
    db.execute("UPDATE rounds SET finished_at = now() WHERE id = %s", (round_id,))
    _, body, _ = call(api, f"/v1/rounds/{round_id}", claims=ADMIN)
    assert (body["status"], body["phase"]) == ("succeeded", None)
    assert body["mlflow_run_url"] == f"{MLFLOW_URL}/#/runs/run"
    assert body["finished_at"] is not None


def test_unknown_round_is_404(api, store, durian):
    status, body, _ = call(api, f"/v1/rounds/{uuid.uuid4()}", claims=ADMIN)
    assert (status, body["error"]["code"]) == (404, "not_found")


def test_round_id_must_be_a_uuid(api, store, durian):
    assert call(api, "/v1/rounds/nope", claims=ADMIN)[0] == 400


# GET /v1/datasets/{name}/versions


@pytest.fixture
def versions(db, durian):
    """v0: images 1-3 (seed labels a, a, b); a round labels image 4 as b -> v1."""
    a, b = _class(db, durian, "a"), _class(db, durian, "b")
    v0 = _version(db, durian, 0)
    images = [_image(db, durian, n, status="labeled") for n in range(1, 4)]
    for image_id, cls in zip(images, (a, a, b), strict=True):
        _in_version(db, v0, image_id, _label(db, image_id, cls))
    round_id = _round(db, durian, status="succeeded")
    new = _image(db, durian, 4, status="labeled")
    v1 = db.execute(
        "INSERT INTO dataset_versions (dataset_id, version, parent_version_id,"
        " created_by_round_id) VALUES (%s, 1, %s, %s) RETURNING id",
        (durian, v0, round_id),
    ).fetchone()[0]
    db.execute(
        "INSERT INTO dataset_version_labels (version_id, image_id, label_id)"
        " SELECT %s, image_id, label_id FROM dataset_version_labels WHERE version_id = %s",
        (v1, v0),
    )
    _in_version(db, v1, new, _label(db, new, b, round_id))
    return {"round_id": round_id, "images": [*images, new]}


def test_versions_newest_first_with_label_counts(api, store, versions):
    status, body, _ = call(api, "/v1/datasets/durian/versions", claims=ADMIN)
    assert status == 200
    assert body["dataset"] == "durian"
    v1, v0 = body["versions"]
    assert {k: v1[k] for k in v1 if k != "created_at"} == {
        "version": 1,
        "parent_version": 0,
        "created_by_round": str(versions["round_id"]),
        "total": 4,
        "label_counts": {"a": 2, "b": 2},
    }
    assert (v0["version"], v0["parent_version"], v0["created_by_round"]) == (0, None, None)
    assert (v0["total"], v0["label_counts"]) == (3, {"a": 2, "b": 1})


def test_versions_of_an_unknown_dataset_is_404(api, store, durian):
    assert call(api, "/v1/datasets/nope/versions", claims=ADMIN)[0] == 404


# GET /v1/datasets/{name}/versions/{n}


def test_version_items_are_paginated(api, store, versions):
    path = "/v1/datasets/durian/versions/1"
    _, first, _ = call(api, path, query={"limit": "3"}, claims=ADMIN)
    assert (first["dataset"], first["version"]) == ("durian", 1)
    assert [i["image_id"] for i in first["items"]] == versions["images"][:3]
    assert first["items"][0] == {
        "image_id": versions["images"][0],
        "s3_key": f"raw/durian/{1:064x}.jpg",
        "split": "train",
        "class_code": "a",
        "label_source": "seed",
        "round_id": None,
    }
    assert first["next_cursor"] == str(versions["images"][2])
    _, rest, _ = call(api, path, query={"limit": "3", "cursor": first["next_cursor"]}, claims=ADMIN)
    assert [i["image_id"] for i in rest["items"]] == versions["images"][3:]
    assert rest["items"][0]["label_source"] == "expert"
    assert rest["items"][0]["round_id"] == str(versions["round_id"])
    assert rest["next_cursor"] is None


def test_a_full_last_page_has_no_next_cursor(api, store, versions):
    _, body, _ = call(api, "/v1/datasets/durian/versions/0", query={"limit": "3"}, claims=ADMIN)
    assert len(body["items"]) == 3
    assert body["next_cursor"] is None


@pytest.mark.parametrize(
    ("path", "query", "status"),
    [
        ("/v1/datasets/durian/versions/9", {}, 404),
        ("/v1/datasets/nope/versions/0", {}, 404),
        ("/v1/datasets/durian/versions/0", {"limit": "1001"}, 400),
        ("/v1/datasets/durian/versions/0", {"limit": "0"}, 400),
        ("/v1/datasets/durian/versions/0", {"cursor": "abc"}, 400),
    ],
)
def test_version_items_errors(api, store, versions, path, query, status):
    assert call(api, path, query=query, claims=ADMIN)[0] == status


def test_data_store_reads_s3_and_a_missing_key_is_none():
    from botocore.stub import Stubber

    from api.deps import DataStore
    from tests.apidb import DATA_BUCKET, fake_s3

    s3 = fake_s3()
    with Stubber(s3) as stub:
        stub.add_client_error("get_object", "NoSuchKey", http_status_code=404)
        assert DataStore(s3, DATA_BUCKET).get("selections/x.json") is None
        stub.add_client_error("get_object", "AccessDenied", http_status_code=403)
        with pytest.raises(s3.exceptions.ClientError):
            DataStore(s3, DATA_BUCKET).get("selections/x.json")
