"""What the routes need from outside: Postgres, S3, settings.

FastAPI dependencies, so tests swap them (`app.dependency_overrides`).

The Lambda runs in the VPC's private subnet next to the EC2 host and reuses
one Postgres connection across requests (a Lambda container serves one
request at a time); a dropped one (the host was stopped) is reopened on the
next request. Presigning S3 URLs is local
signing with the Lambda's credentials, with no AWS call; the few S3 reads
(a round's selection) go through the VPC's S3 gateway endpoint. The Lambda
needs no route to the internet.
"""

import base64
import os
from dataclasses import dataclass
from functools import cache
from typing import Annotated

import boto3
import psycopg
from botocore.config import Config
from fastapi import Depends

from lambdas.postgres import open_db

# Lifetime of every presigned download and upload link.
URL_EXPIRES_IN = 900

_conn: psycopg.Connection | None = None


def get_db() -> psycopg.Connection:
    global _conn
    if _conn is None or _conn.closed:
        _conn = open_db()
    return _conn


def signing_config() -> Config:
    # SigV4 on the regional host: the default for presigning is the old
    # signature on s3.amazonaws.com, which redirects for buckets outside
    # us-east-1 (a redirected PUT fails).
    return Config(signature_version="s3v4", s3={"addressing_style": "virtual"})


@cache
def get_s3():
    return boto3.client("s3", region_name=os.environ["AWS_REGION"], config=signing_config())


@dataclass(frozen=True)
class Settings:
    models_bucket: str
    data_bucket: str
    # Captures one uploader may send in any 24 hours (design doc, sign in to contribute).
    upload_daily_limit: int
    label_studio_url: str
    mlflow_url: str


@cache
def get_settings() -> Settings:
    return Settings(
        models_bucket=os.environ["MODELS_BUCKET"],
        data_bucket=os.environ["DATA_BUCKET"],
        upload_daily_limit=int(os.environ["UPLOAD_DAILY_LIMIT"]),
        label_studio_url=os.environ["LABEL_STUDIO_URL"],
        mlflow_url=os.environ["MLFLOW_URL"],
    )


def presign_get(s3, bucket: str, key: str) -> str:
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=URL_EXPIRES_IN
    )


def presign_put(
    s3, bucket: str, key: str, *, content_type: str, size: int, sha256: str
) -> tuple[str, dict[str, str]]:
    """A PUT link and the headers the upload must send. Type, length and
    SHA-256 are signed, so S3 refuses a body that differs from what was
    declared (a wrong size, or another file under this sha256 key)."""
    checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
    url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ContentType": content_type,
            "ContentLength": size,
            "ChecksumSHA256": checksum,
        },
        ExpiresIn=URL_EXPIRES_IN,
    )
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(size),
        "x-amz-checksum-sha256": checksum,
    }
    return url, headers


class DataStore:
    """Reads small files from the data bucket (a round's selection)."""

    def __init__(self, s3, bucket: str) -> None:
        self.s3 = s3
        self.bucket = bucket

    def get(self, key: str) -> bytes | None:
        # A missing key is NoSuchKey only with s3:ListBucket; without it S3 says 403.
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None


def get_data_store(
    s3: Annotated[object, Depends(get_s3)], settings: Annotated[Settings, Depends(get_settings)]
) -> DataStore:
    return DataStore(s3, settings.data_bucket)
