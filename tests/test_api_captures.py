"""POST /v1/captures: the app sends its prediction and gets a presigned PUT for
the image (design doc F1 steps 2 to 5, report section 10.3).

Only `uploaders` may send; each may send UPLOAD_LIMIT captures in any 24 hours.
The capture_id is the retry key. Against a real Postgres (TEST_DATABASE_URL);
the upload link is signed locally with fake credentials.
"""

import base64
import hashlib
import uuid

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mangum")

from tests.apidb import (
    DATA_BUCKET,
    REGION,
    UPLOAD_LIMIT,
    crop,
    dataset,
    model,
    presigned,
)
from tests.apigw import call

UPLOADER = {"sub": "sub-1", "username": "google_1", "cognito:groups": "[uploaders]"}
OTHER = {"sub": "sub-2", "username": "google_2", "cognito:groups": "[uploaders]"}
IMAGE = b"\xff\xd8\xff\xe0 not really a jpeg"
IMAGE_SHA = hashlib.sha256(IMAGE).hexdigest()


@pytest.fixture
def leaf(db):
    """Dataset `durian` with model durian-fastvit/3 (classes a, b), and an
    `other` dataset with its own model."""
    durian = dataset(db, "durian", crop(db, "durian", "Sầu riêng"), "Lá sầu riêng")
    other = dataset(db, "other", crop(db, "other", "Khác"), "Khác")
    return {
        "durian": durian,
        "model": model(db, durian, "durian-fastvit", 3),
        "other_model": model(db, other, "other-fastvit", 1),
    }


def _body(**overrides) -> dict:
    body = {
        "capture_id": str(uuid.uuid4()),
        "dataset": "durian",
        "captured_at": "2026-10-13T08:12:00+07:00",
        "image": {"sha256": IMAGE_SHA, "content_type": "image/jpeg", "size": len(IMAGE)},
        "inference": {"model_version": "durian-fastvit/3", "probs": {"a": 0.3, "b": 0.7}},
    }
    body.update(overrides)
    return body


def _send(api, body: dict, claims: dict | None = UPLOADER):
    return call(api, "/v1/captures", method="POST", body=body, claims=claims)


def _rows(db) -> list[tuple]:
    return db.execute(
        "SELECT c.capture_id, c.user_sub, c.sha256, c.image_id, cl.code, c.confidence,"
        " c.probs, c.model_version_id, c.content_type, c.size_bytes"
        " FROM captures c JOIN classes cl ON cl.id = c.top1_class_id ORDER BY c.id"
    ).fetchall()


# Who may send


def test_needs_sign_in(api, leaf):
    status, body, _ = _send(api, _body(), claims=None)
    assert (status, body["error"]["code"]) == (401, "unauthenticated")


def test_needs_the_uploaders_group(api, leaf):
    labeler = {"sub": "sub-9", "cognito:groups": "[labelers admins]"}
    status, body, _ = _send(api, _body(), claims=labeler)
    assert (status, body["error"]["code"]) == (403, "forbidden")


def test_the_uploader_comes_from_the_token_not_the_body(api, db, leaf):
    status, body, _ = _send(api, _body(user_sub="someone-else"))
    assert (status, body["error"]["code"]) == (400, "invalid_request")
    assert _rows(db) == []


# A new image


def test_new_image_gets_a_presigned_put(api, db, leaf):
    request = _body()
    status, body, _ = _send(api, request)
    assert status == 200
    assert body["capture_id"] == request["capture_id"]
    assert body["status"] == "upload_required"
    upload = body["upload"]
    assert upload["method"] == "PUT"
    assert upload["expires_in"] == 900
    checksum = base64.b64encode(hashlib.sha256(IMAGE).digest()).decode()
    # The PUT must send exactly these; S3 rejects a body of another size or
    # SHA-256, so incoming/{sha256}.jpg always holds what the app declared.
    assert upload["headers"] == {
        "Content-Type": "image/jpeg",
        "Content-Length": str(len(IMAGE)),
        "x-amz-checksum-sha256": checksum,
    }
    host, path, query = presigned(upload["url"])
    assert host == f"{DATA_BUCKET}.s3.{REGION}.amazonaws.com"
    assert path == f"/incoming/{IMAGE_SHA}.jpg"
    assert query["X-Amz-Expires"] == "900"
    assert query["X-Amz-SignedHeaders"] == "content-length;content-type;host;x-amz-checksum-sha256"

    (row,) = _rows(db)
    assert row == (
        uuid.UUID(request["capture_id"]),
        "sub-1",
        IMAGE_SHA,
        None,  # set by the validation Lambda once the upload checks out
        "b",
        pytest.approx(0.7),
        {"a": 0.3, "b": 0.7},
        leaf["model"],
        "image/jpeg",
        len(IMAGE),
    )


def test_png_is_signed_as_png(api, leaf):
    image = {"sha256": IMAGE_SHA, "content_type": "image/png", "size": 10}
    _, body, _ = _send(api, _body(image=image))
    assert body["upload"]["headers"]["Content-Type"] == "image/png"


def test_known_image_needs_no_upload(api, db, leaf):
    image_id = db.execute(
        "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
        " VALUES (%s, %s, %s, 'unlabeled', 'train', 'edge') RETURNING id",
        (leaf["durian"], IMAGE_SHA, f"raw/durian/{IMAGE_SHA}.jpg"),
    ).fetchone()[0]
    status, body, _ = _send(api, _body())
    assert status == 200
    assert body["status"] == "already_exists"
    assert body.get("upload") is None
    assert _rows(db)[0][3] == image_id


def test_the_same_image_in_another_dataset_is_new(api, db, leaf):
    db.execute(
        "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
        " SELECT id, %s, 'k', 'unlabeled', 'train', 'edge' FROM datasets WHERE name = 'other'",
        (IMAGE_SHA,),
    )
    assert _send(api, _body())[1]["status"] == "upload_required"


# Retries


def test_a_retry_returns_the_same_answer_and_stores_one_row(api, db, leaf):
    request = _body()
    first = _send(api, request)[1]
    again = _send(api, request)[1]
    assert first["status"] == again["status"] == "upload_required"
    assert again["upload"]["headers"] == first["upload"]["headers"]
    assert presigned(again["upload"]["url"])[1] == presigned(first["upload"]["url"])[1]
    assert len(_rows(db)) == 1


def test_a_retry_after_the_image_arrived_needs_no_upload(api, db, leaf):
    request = _body()
    _send(api, request)
    db.execute(
        "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
        " VALUES (%s, %s, 'k', 'unlabeled', 'train', 'edge')",
        (leaf["durian"], IMAGE_SHA),
    )
    assert _send(api, request)[1]["status"] == "already_exists"


def test_a_retry_with_another_offset_for_the_same_instant_is_the_same(api, db, leaf):
    request = _body()
    _send(api, request)
    status, _, _ = _send(api, {**request, "captured_at": "2026-10-13T01:12:00Z"})
    assert status == 200
    assert len(_rows(db)) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": "other"},
        {"captured_at": "2026-10-13T08:13:00+07:00"},
        {"image": {"sha256": "cd" * 32, "content_type": "image/jpeg", "size": len(IMAGE)}},
        {"image": {"sha256": IMAGE_SHA, "content_type": "image/png", "size": len(IMAGE)}},
        {"image": {"sha256": IMAGE_SHA, "content_type": "image/jpeg", "size": 99}},
        {"inference": {"model_version": "durian-fastvit/3", "probs": {"a": 0.4, "b": 0.6}}},
    ],
)
def test_the_same_capture_id_with_another_body_is_409(api, db, leaf, change):
    request = _body()
    _send(api, request)
    status, body, _ = _send(api, {**request, **change})
    assert (status, body["error"]["code"]) == (409, "conflict")
    assert len(_rows(db)) == 1


def test_another_users_capture_id_is_409(api, db, leaf):
    request = _body()
    _send(api, request)
    status, _, _ = _send(api, request, claims=OTHER)
    assert status == 409


# What the API checks


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"dataset": "nope"}, 404),
        ({"inference": {"model_version": "durian-fastvit/9", "probs": {"a": 0.5, "b": 0.5}}}, 404),
        # A model of another dataset.
        ({"inference": {"model_version": "other-fastvit/1", "probs": {"a": 0.5, "b": 0.5}}}, 404),
        ({"inference": {"model_version": "durian-fastvit", "probs": {"a": 0.5, "b": 0.5}}}, 400),
        # probs keys must be the model's classes.
        ({"inference": {"model_version": "durian-fastvit/3", "probs": {"a": 0.5, "c": 0.5}}}, 400),
        (
            {
                "inference": {
                    "model_version": "durian-fastvit/3",
                    "probs": {"a": 0.4, "b": 0.4, "c": 0.2},
                }
            },
            400,
        ),
        ({"inference": {"model_version": "durian-fastvit/3", "probs": {"a": 0.5, "b": 0.6}}}, 400),
        ({"image": {"sha256": IMAGE_SHA, "content_type": "image/jpeg", "size": 11 << 20}}, 400),
        ({"image": {"sha256": IMAGE_SHA, "content_type": "image/gif", "size": 10}}, 400),
    ],
)
def test_bad_captures_are_refused_and_not_stored(api, db, leaf, change, status):
    got, body, _ = _send(api, _body(**change))
    assert got == status
    assert body["error"]["code"] == {400: "invalid_request", 404: "not_found"}[status]
    assert _rows(db) == []


# The daily limit


def _fill(api, claims=UPLOADER) -> None:
    for _ in range(UPLOAD_LIMIT):
        assert _send(api, _body(), claims=claims)[0] == 200


def test_over_the_limit_is_429_with_retry_after(api, db, leaf):
    _fill(api)
    status, body, headers = _send(api, _body())
    assert (status, body["error"]["code"]) == (429, "quota_exceeded")
    assert 86_000 < int(headers["retry-after"]) <= 86_400
    assert len(_rows(db)) == UPLOAD_LIMIT


def test_the_limit_is_per_user(api, leaf):
    _fill(api)
    assert _send(api, _body(), claims=OTHER)[0] == 200


def test_retry_after_is_when_the_oldest_capture_leaves_the_window(api, db, leaf):
    _fill(api)
    db.execute(
        "UPDATE captures SET created_at = now() - interval '23 hours'"
        " WHERE id = (SELECT min(id) FROM captures)"
    )
    retry_after = int(_send(api, _body())[2]["retry-after"])
    assert 3_500 < retry_after <= 3_600


def test_captures_older_than_24_hours_do_not_count(api, db, leaf):
    _fill(api)
    db.execute("UPDATE captures SET created_at = now() - interval '25 hours'")
    assert _send(api, _body())[0] == 200


def test_a_retry_is_answered_over_the_limit(api, db, leaf):
    request = _body()
    assert _send(api, request)[0] == 200
    for _ in range(UPLOAD_LIMIT - 1):
        _send(api, _body())
    assert _send(api, _body())[0] == 429
    assert _send(api, request)[0] == 200


def test_parallel_requests_cannot_pass_the_limit_together(db, pg_url, leaf):
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import HTTPException

    from api.auth import User
    from api.routes.captures import create_capture
    from common.edge import CaptureRequest
    from db import connect
    from tests.apidb import SETTINGS, fake_s3

    s3, user = fake_s3(), User(sub="sub-1", username="google_1", groups=["uploaders"])

    def send(_) -> int:
        with connect(pg_url) as conn:
            try:
                create_capture(CaptureRequest(**_body()), user, conn, s3, SETTINGS)
            except HTTPException as e:
                return e.status_code
        return 200

    with ThreadPoolExecutor(8) as pool:
        statuses = sorted(pool.map(send, range(8)))
    assert statuses == [200] * UPLOAD_LIMIT + [429] * (8 - UPLOAD_LIMIT)
    assert len(_rows(db)) == UPLOAD_LIMIT
