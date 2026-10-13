"""Upload validation (design doc F1 step 6): incoming/ -> raw/ and the pool.

The app PUTs an image to incoming/{sha256}.jpg with the presigned link from
POST /v1/captures; S3 signed its size and SHA-256, so the key names the
bytes. S3 notifies an SQS queue and this Lambda (in the VPC: Postgres, and S3
through the gateway endpoint) takes each new object:

1. Reads it and checks the bytes hash to the key's sha256.
2. Finds the captures still waiting for that image (image_id null), per
   dataset. None: the object is an orphan and is deleted.
3. Decodes it with Pillow: a whole JPEG or PNG of the declared type, at most
   MAX_PIXELS. Turns it the way the phone showed it (EXIF orientation), then
   re-encodes it as a JPEG without EXIF, so no GPS or camera data is kept.
4. For each dataset: writes raw/{dataset}/{sha256}.jpg and adds the image to
   the train pool (split train, status unlabeled, source edge), then links
   the captures. An image the dataset already has (e.g. from the seed) keeps
   its row, split and file; only the captures are linked.
5. Deletes the incoming/ object.

A refused upload is deleted and its captures stay unlinked; nothing reaches
the pool. Every step can run again: the queue redelivers a message that
failed (for example while the EC2 host is stopped), and a run that died
half way finishes on the next delivery.
"""

import hashlib
import io
import json
import logging
import os
import re
from urllib.parse import unquote_plus

import boto3
import psycopg
from PIL import Image, ImageOps, UnidentifiedImageError

from common.edge import MAX_UPLOAD_BYTES
from common.s3keys import raw_key
from lambdas.postgres import open_db

log = logging.getLogger()
log.setLevel(logging.INFO)

# 40 megapixels: more than any phone photo, far below what would exhaust memory.
MAX_PIXELS = 40_000_000
JPEG_QUALITY = 95
FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png"}

_KEY = re.compile(r"^incoming/([0-9a-f]{64})\.jpg$")

_WAITING = """
    SELECT c.dataset_id, d.name, array_agg(DISTINCT c.content_type)
      FROM captures c
      JOIN datasets d ON d.id = c.dataset_id
     WHERE c.sha256 = %s AND c.image_id IS NULL
     GROUP BY c.dataset_id, d.name
     ORDER BY c.dataset_id
"""


class InvalidImage(ValueError):
    pass


class S3Store:
    def __init__(self, bucket: str, client=None) -> None:
        self.bucket = bucket
        self.client = client or boto3.client("s3")

    def get(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def put_jpeg(self, key: str, body: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="image/jpeg")

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


def clean_image(data: bytes) -> tuple[str, bytes]:
    """The image's type and a clean JPEG of it; InvalidImage if it is not a
    whole JPEG or PNG within the size limits."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImage(f"{len(data)} bytes is over the upload limit")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            kind = FORMATS.get(probe.format)
            if kind is None:
                raise InvalidImage(f"{probe.format} is not JPEG or PNG")
            width, height = probe.size
            if width * height > MAX_PIXELS:
                raise InvalidImage(f"{width}x{height} has too many pixels")
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            upright = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise InvalidImage(str(e) or type(e).__name__) from e
    out = io.BytesIO()
    upright.save(out, "JPEG", quality=JPEG_QUALITY)  # no exif= : nothing is kept
    return kind, out.getvalue()


def _has_image(conn: psycopg.Connection, dataset_id: int, sha256: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM images WHERE dataset_id = %s AND sha256 = %s", (dataset_id, sha256)
    ).fetchone()
    return row is not None


def _add_to_pool(conn: psycopg.Connection, dataset_id: int, sha256: str, key: str, kind: str):
    with conn.transaction():
        conn.execute(
            "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
            " VALUES (%s, %s, %s, 'unlabeled', 'train', 'edge')"
            " ON CONFLICT (dataset_id, sha256) DO NOTHING",
            (dataset_id, sha256, key),
        )
        conn.execute(
            "UPDATE captures SET image_id ="
            " (SELECT id FROM images WHERE dataset_id = %s AND sha256 = %s)"
            " WHERE dataset_id = %s AND sha256 = %s AND image_id IS NULL"
            " AND content_type = %s",
            (dataset_id, sha256, dataset_id, sha256, kind),
        )


def process(conn: psycopg.Connection, store, key: str) -> str:
    """Handle one incoming/ object; returns what happened."""
    match = _KEY.match(key)
    if not match:
        return "ignored"
    sha256 = match.group(1)
    data = store.get(key)
    if data is None:
        return "gone"  # handled by an earlier delivery

    def reject(reason: str) -> str:
        log.warning("refused %s: %s", key, reason)
        store.delete(key)
        return "rejected"

    if hashlib.sha256(data).hexdigest() != sha256:
        return reject("bytes do not match the key")
    waiting = conn.execute(_WAITING, (sha256,)).fetchall()
    if not waiting:
        return reject("no capture is waiting for it")
    try:
        kind, jpeg = clean_image(data)
    except InvalidImage as e:
        return reject(str(e))
    targets = [(ds, name) for ds, name, declared in waiting if kind in declared]
    if not targets:
        return reject(f"it is {kind}, not the declared type")

    for dataset_id, name in targets:
        target = raw_key(name, sha256)
        if not _has_image(conn, dataset_id, sha256):  # else its file is in place
            store.put_jpeg(target, jpeg)
        _add_to_pool(conn, dataset_id, sha256, target, kind)
    store.delete(key)
    log.info("stored %s for %s", key, ", ".join(name for _, name in targets))
    return "stored"


_conn: psycopg.Connection | None = None


def _db() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed:
        _conn = open_db()
    return _conn


def _store() -> S3Store:
    return S3Store(os.environ["DATA_BUCKET"])


def handler(event: dict, _context) -> dict:
    """SQS batch of S3 notifications; failed messages go back to the queue."""
    failures = []
    for message in event["Records"]:
        try:
            body = json.loads(message["body"])
            for record in body.get("Records", []):  # none in S3's test event
                process(_db(), _store(), unquote_plus(record["s3"]["object"]["key"]))
        except Exception:
            log.exception("message %s failed; it will be retried", message["messageId"])
            failures.append({"itemIdentifier": message["messageId"]})
    return {"batchItemFailures": failures}
