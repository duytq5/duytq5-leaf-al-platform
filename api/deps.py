"""What the routes need from outside: Postgres, S3 signing, settings.

FastAPI dependencies, so tests swap them (`app.dependency_overrides`).

The Lambda runs in the VPC's private subnet next to the EC2 host and reuses
one Postgres connection across requests (a Lambda container serves one
request at a time); a dropped one (the host was stopped) is reopened on the
next request. Presigning S3 URLs is local
signing with the Lambda's credentials, with no AWS call, so the Lambda needs
no route to the internet.
"""

import os
from dataclasses import dataclass
from functools import cache

import boto3
import psycopg
from botocore.config import Config

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


@cache
def get_settings() -> Settings:
    return Settings(models_bucket=os.environ["MODELS_BUCKET"])


def presign_get(s3, bucket: str, key: str) -> str:
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=URL_EXPIRES_IN
    )
