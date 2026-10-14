"""The viewer's sync service: every few minutes, copy new raw/ images from S3
and refresh one FiftyOne dataset per platform dataset from Postgres. The
FiftyOne app (a separate container) reads the same MongoDB.

Environment (deploy/ec2/docker-compose.yml): DATABASE_URL, DATA_BUCKET,
AWS_REGION, MEDIA_DIR, SYNC_MINUTES, FIFTYONE_DATABASE_URI.
"""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import boto3
import psycopg

from db import connect
from viewer.media import sync_media
from viewer.records import ImageRecord, dataset_names, load_records

log = logging.getLogger("viewer")

Publish = Callable[[str, list[tuple[ImageRecord, Path]]], None]


def sync_once(
    conn: psycopg.Connection,
    s3,
    bucket: str,
    media_root: Path,
    *,
    publish: Publish,
    media=sync_media,
) -> dict[str, int]:
    """Publish every dataset with the images that have a local copy; return
    how many images each one shows."""
    counts = {}
    for name in dataset_names(conn):
        records = load_records(conn, name)
        paths = media(s3, bucket, [r.s3_key for r in records], media_root)
        pairs = [(r, paths[r.s3_key]) for r in records if r.s3_key in paths]
        publish(name, pairs)
        counts[name] = len(pairs)
    return counts


def _publish(name: str, pairs: list[tuple[ImageRecord, Path]]) -> None:
    from viewer.samples import sync_dataset, to_sample

    sync_dataset(name, [to_sample(record, path) for record, path in pairs])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bucket = os.environ["DATA_BUCKET"]
    media_root = Path(os.environ.get("MEDIA_DIR", "/media"))
    every = 60 * float(os.environ.get("SYNC_MINUTES", "10"))
    s3 = boto3.client("s3", region_name=os.environ["AWS_REGION"])
    while True:
        try:
            with connect() as conn:
                counts = sync_once(conn, s3, bucket, media_root, publish=_publish)
            log.info("synced %s", counts)
        except Exception:  # Postgres restarting, S3 hiccup: try again next time
            log.exception("sync failed")
        time.sleep(every)


if __name__ == "__main__":
    main()
