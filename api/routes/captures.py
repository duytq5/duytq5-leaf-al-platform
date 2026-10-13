"""POST /v1/captures (design doc F1, report section 10.3).

The app sends its on-device prediction; the API stores a `captures` row and
answers with a presigned PUT for incoming/{sha256}.jpg, or `already_exists`
when the dataset has that image. The image bytes never pass through Lambda;
the upload validation Lambda later moves the file to raw/ and links the row.

capture_id is the retry key: the same id and body again gets the same kind of
answer (`already_exists` once the image has arrived) and is never counted
twice; the same id with another body, or from another user, is a 409. Each
uploader may send `upload_daily_limit` new captures in any 24 hours (429 with
Retry-After); requests of one user are serialized, so parallel requests
cannot pass the limit together.
"""

import math
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from psycopg.types.json import Jsonb

from api.auth import User, require_groups
from api.deps import URL_EXPIRES_IN, Settings, get_db, get_s3, get_settings, presign_put
from common.edge import CaptureRequest, CaptureResponse, CaptureStatus, UploadInstruction
from common.s3keys import incoming_key

router = APIRouter()

WINDOW_SECONDS = 24 * 3600

_EXISTING = """
    SELECT c.user_sub, d.name, c.sha256, c.content_type, c.size_bytes,
           mv.mlflow_name || '/' || mv.mlflow_version, c.probs, c.captured_at, c.dataset_id
      FROM captures c
      JOIN datasets d ON d.id = c.dataset_id
      JOIN model_versions mv ON mv.id = c.model_version_id
     WHERE c.capture_id = %s
"""

_MODEL = """
    SELECT mv.id, mv.labels
      FROM model_versions mv
      JOIN rounds r ON r.id = mv.round_id
     WHERE mv.mlflow_name = %s AND mv.mlflow_version = %s AND r.dataset_id = %s
"""

_RECENT = """
    SELECT extract(epoch FROM created_at + make_interval(secs => %s) - now())
      FROM captures
     WHERE user_sub = %s AND created_at > now() - make_interval(secs => %s)
     ORDER BY created_at
"""

_INSERT = """
    INSERT INTO captures (capture_id, image_id, dataset_id, sha256, user_sub,
                          model_version_id, top1_class_id, confidence, probs,
                          captured_at, content_type, size_bytes)
    VALUES (%s, %s, %s, %s, %s, %s,
            (SELECT id FROM classes WHERE dataset_id = %s AND code = %s),
            %s, %s, %s, %s, %s)
"""


def _image_id(db: psycopg.Connection, dataset_id: int, sha256: str) -> int | None:
    row = db.execute(
        "SELECT id FROM images WHERE dataset_id = %s AND sha256 = %s", (dataset_id, sha256)
    ).fetchone()
    return row[0] if row else None


def _check_retry(row: tuple, body: CaptureRequest, user: User) -> int:
    """The dataset id of a capture sent before, if this is the same request."""
    user_sub, dataset, sha256, content_type, size, model_version, probs, captured_at, ds = row
    same = (
        user_sub == user.sub
        and dataset == body.dataset
        and sha256 == body.image.sha256
        and content_type == body.image.content_type
        and size == body.image.size
        and model_version == body.inference.model_version
        and probs == body.inference.probs
        and captured_at == body.captured_at
    )
    if not same:
        raise HTTPException(409, "This capture_id was already sent with another body.")
    return ds


def _check_limit(db: psycopg.Connection, user: User, limit: int) -> None:
    # Seconds until each capture of the last 24 hours leaves the window, oldest first.
    left = [r[0] for r in db.execute(_RECENT, (WINDOW_SECONDS, user.sub, WINDOW_SECONDS))]
    if len(left) >= limit:
        # A slot frees when the newest capture beyond the limit leaves.
        retry_after = max(1, math.ceil(left[len(left) - limit]))
        raise HTTPException(
            429,
            f"Daily limit reached: {limit} images in 24 hours.",
            headers={"Retry-After": str(retry_after)},
        )


def _new_capture(db: psycopg.Connection, body: CaptureRequest, user: User, limit: int) -> int:
    """Store a new capture after the checks; returns its dataset id."""
    row = db.execute("SELECT id FROM datasets WHERE name = %s", (body.dataset,)).fetchone()
    if row is None:
        raise HTTPException(404, f"No dataset named {body.dataset}.")
    dataset_id = row[0]
    model, version = body.inference.model_version.rsplit("/", 1)
    row = db.execute(_MODEL, (model, int(version), dataset_id)).fetchone()
    if row is None:
        raise HTTPException(
            404, f"No model {body.inference.model_version} for dataset {body.dataset}."
        )
    model_version_id, labels = row
    probs = body.inference.probs
    codes = {label["code"] for label in labels}
    if set(probs) != codes:
        raise HTTPException(
            400, f"inference.probs keys must be the model's classes: {', '.join(sorted(codes))}."
        )
    _check_limit(db, user, limit)
    top1 = max(probs, key=probs.__getitem__)
    db.execute(
        _INSERT,
        (
            body.capture_id,
            _image_id(db, dataset_id, body.image.sha256),
            dataset_id,
            body.image.sha256,
            user.sub,
            model_version_id,
            dataset_id,
            top1,
            probs[top1],
            Jsonb(probs),
            body.captured_at,
            body.image.content_type,
            body.image.size,
        ),
    )
    return dataset_id


@router.post("/v1/captures")
def create_capture(
    body: CaptureRequest,
    user: Annotated[User, Depends(require_groups("uploaders"))],
    db: Annotated[psycopg.Connection, Depends(get_db)],
    s3: Annotated[object, Depends(get_s3)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CaptureResponse:
    with db.transaction():
        # One request per user at a time, for the daily limit.
        db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (user.sub,))
        row = db.execute(_EXISTING, (body.capture_id,)).fetchone()
        if row is not None:
            dataset_id = _check_retry(row, body, user)
        else:
            dataset_id = _new_capture(db, body, user, settings.upload_daily_limit)
        known = _image_id(db, dataset_id, body.image.sha256) is not None

    if known:
        return CaptureResponse(capture_id=body.capture_id, status=CaptureStatus.ALREADY_EXISTS)
    url, headers = presign_put(
        s3,
        settings.data_bucket,
        incoming_key(body.image.sha256),
        content_type=body.image.content_type,
        size=body.image.size,
        sha256=body.image.sha256,
    )
    return CaptureResponse(
        capture_id=body.capture_id,
        status=CaptureStatus.UPLOAD_REQUIRED,
        upload=UploadInstruction(url=url, headers=headers, expires_in=URL_EXPIRES_IN),
    )
