"""Rows and settings for the API tests (tests/conftest.py has the `db` and
`api` fixtures)."""

import uuid
from urllib.parse import parse_qs, urlsplit

import boto3

from api.deps import Settings, signing_config

REGION = "ap-southeast-2"
MODELS_BUCKET = "leaf-al-models"
DATA_BUCKET = "leaf-al-data"
UPLOAD_LIMIT = 3
LABEL_STUDIO_URL = "https://label.example.com"
MLFLOW_URL = "http://localhost:5000"
SETTINGS = Settings(
    models_bucket=MODELS_BUCKET,
    data_bucket=DATA_BUCKET,
    upload_daily_limit=UPLOAD_LIMIT,
    label_studio_url=LABEL_STUDIO_URL,
    mlflow_url=MLFLOW_URL,
)
SHA = "ab" * 32


def fake_s3():
    """Signs like the Lambda's client; signing makes no AWS call."""
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id="AKIDTEST",
        aws_secret_access_key="secret",
        config=signing_config(),
    )


CLASSES = ("a", "b")


def crop(conn, code: str, display_name: str, description: str | None = None) -> int:
    return conn.execute(
        "INSERT INTO crops (code, display_name, description) VALUES (%s, %s, %s) RETURNING id",
        (code, display_name, description),
    ).fetchone()[0]


def dataset(conn, name: str, crop_id: int, display_name: str) -> int:
    dataset_id = conn.execute(
        "INSERT INTO datasets (name, crop_id, display_name) VALUES (%s, %s, %s) RETURNING id",
        (name, crop_id, display_name),
    ).fetchone()[0]
    conn.execute("INSERT INTO dataset_versions (dataset_id, version) VALUES (%s, 0)", (dataset_id,))
    for position, code in enumerate(CLASSES):
        conn.execute(
            "INSERT INTO classes (dataset_id, code, position, display_name)"
            " VALUES (%s, %s, %s, %s)",
            (dataset_id, code, position, code.upper()),
        )
    return dataset_id


def model(conn, dataset_id: int, name: str, version: int, exported: bool = True) -> int:
    """A model version trained by its own (finished) round."""
    round_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, strategy, k, seed,"
        " status, started_by, commit_sha)"
        " SELECT %s, %s, id, 'random', 5, 0, 'succeeded', 'duytq5', %s"
        " FROM dataset_versions WHERE dataset_id = %s AND version = 0",
        (round_id, dataset_id, "c" * 40, dataset_id),
    )
    export = (
        (f"edge/{name}/{version}/model.onnx", SHA, 1000, "now()")
        if exported
        else (None, None, None, None)
    )
    return conn.execute(
        "INSERT INTO model_versions (round_id, trained_on_version_id, mlflow_name,"
        " mlflow_version, mlflow_run_id, arch, train_config, labels,"
        " onnx_s3_key, onnx_sha256, size_bytes, exported_at)"
        " SELECT %s, base_version_id, %s, %s, 'run', 'fastvit_t8', '{}',"
        """ '[{"code": "a", "display_name": "A"}, {"code": "b", "display_name": "B"}]',"""
        " %s, %s, %s, %s::timestamptz FROM rounds WHERE id = %s RETURNING id",
        (round_id, name, version, *export, round_id),
    ).fetchone()[0]


def release(conn, model_version_id: int, action: str = "promote") -> None:
    conn.execute(
        "INSERT INTO model_releases (model_version_id, action, released_by, commit_sha)"
        " VALUES (%s, %s, 'duytq5', %s)",
        (model_version_id, action, uuid.uuid4().hex + "0" * 8),
    )


def make_catalog(conn) -> dict[str, int]:
    """Coffee (RoCoLe, released twice), durian (leaf released, fruit not),
    and a crop with no dataset."""
    coffee = crop(conn, "coffee", "Cà phê")
    durian = crop(conn, "durian", "Sầu riêng", "Cây ăn quả")
    crop(conn, "pepper", "Hồ tiêu")
    rocole = dataset(conn, "rocole", coffee, "Lá cà phê (RoCoLe)")
    leaf = dataset(conn, "durian", durian, "Lá sầu riêng")
    dataset(conn, "durian_fruit", durian, "Quả sầu riêng")
    models = {
        "rocole/1": model(conn, rocole, "rocole-fastvit", 1),
        "rocole/2": model(conn, rocole, "rocole-fastvit", 2),
        "durian/1": model(conn, leaf, "durian-fastvit", 1),
    }
    release(conn, models["rocole/1"])
    release(conn, models["rocole/2"])
    release(conn, models["durian/1"])
    return models


def presigned(url: str) -> tuple[str, str, dict[str, str]]:
    """Host, path and query parameters of a presigned URL."""
    parts = urlsplit(url)
    return parts.netloc, parts.path, {k: v[0] for k, v in parse_qs(parts.query).items()}
