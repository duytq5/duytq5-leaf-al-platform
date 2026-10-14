"""Local copies of raw/ images: FiftyOne (open source) shows local files only.

Keys are raw/{dataset}/{sha256}.jpg, so a file that exists never changes and
is not downloaded again."""

import logging
from pathlib import Path

from botocore.exceptions import ClientError

log = logging.getLogger(__name__)


def sync_media(s3, bucket: str, keys: list[str], root: Path) -> dict[str, Path]:
    """Download the keys not under `root` yet; return the local path of every
    key that is there. A key missing in S3 is logged and left out."""
    paths: dict[str, Path] = {}
    for key in keys:
        path = root / key
        if not path.exists():
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                    raise
                log.warning("missing in S3: %s", key)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            part.write_bytes(body)
            part.replace(path)  # never a half-written image
        paths[key] = path
    return paths
