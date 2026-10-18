"""Repair app images stored before the upload fix.

The upload Lambda re-encodes every app image (to drop EXIF) but used to store
the result under the upload's sha256: raw/{dataset}/{upload sha256}.jpg, with
that hash on the images row. The worker checks every raw/ file against
images.sha256, so a round that scored or trained on such an image failed
with "does not match its sha256".

For each app image (source 'edge') whose file does not hash to its row, this
copies the file to raw/{dataset}/{file sha256}.jpg and points the row at it.
The old file stays, since earlier Label Studio tasks may still open it. Seed
images are never touched. Running it again finds nothing to do.
"""

import hashlib
from dataclasses import dataclass

import psycopg

from common.s3keys import raw_key


@dataclass(frozen=True)
class Fix:
    image_id: int
    dataset: str
    old_sha: str
    new_sha: str
    old_key: str

    @property
    def new_key(self) -> str:
        return raw_key(self.dataset, self.new_sha)


class S3Store:
    def __init__(self, bucket: str, client=None) -> None:
        import boto3

        self.bucket = bucket
        self.client = client or boto3.client("s3")

    def get(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def put_jpeg(self, key: str, body: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="image/jpeg")


def plan(conn: psycopg.Connection, store) -> list[Fix]:
    """The app images whose file does not hash to their row; LookupError if a file is missing."""
    rows = conn.execute(
        """
        SELECT i.id, d.name, i.sha256, i.s3_key
          FROM images i JOIN datasets d ON d.id = i.dataset_id
         WHERE i.source = 'edge'
         ORDER BY i.id
        """
    ).fetchall()
    fixes = []
    for image_id, name, sha, key in rows:
        body = store.get(key)
        if body is None:
            raise LookupError(f"image {image_id}: no file at {key}")
        actual = hashlib.sha256(body).hexdigest()
        if actual != sha:
            fixes.append(Fix(image_id, name, sha, actual, key))
    return fixes


def apply(conn: psycopg.Connection, store, fixes: list[Fix]) -> None:
    for fix in fixes:
        body = store.get(fix.old_key)
        if body is None or hashlib.sha256(body).hexdigest() != fix.new_sha:
            raise LookupError(f"image {fix.image_id}: {fix.old_key} changed since the plan")
        store.put_jpeg(fix.new_key, body)
        with conn.transaction():
            conn.execute(
                "UPDATE images SET sha256 = %s, s3_key = %s WHERE id = %s AND sha256 = %s",
                (fix.new_sha, fix.new_key, fix.image_id, fix.old_sha),
            )
