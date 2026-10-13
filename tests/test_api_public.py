"""Public app routes: GET /v1/crops and GET /v1/devices/config.

No token: API Gateway passes no claims. The current model of a dataset is its
latest model_releases row (design doc F5), against a real Postgres
(TEST_DATABASE_URL). Download links are presigned locally by botocore with
fake credentials, so no AWS call is made.
"""

import uuid
from urllib.parse import parse_qs, urlsplit

import psycopg
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mangum")

import boto3

from api import deps
from api.app import app, handler
from api.deps import Settings, get_db, get_s3, get_settings, signing_config
from db import connect
from db.migrate import migrate
from tests.apigw import call

REGION = "ap-southeast-2"
MODELS_BUCKET = "leaf-al-models"
SHA = "ab" * 32


@pytest.fixture
def conn(pg_url):
    with connect(pg_url) as c:
        migrate(c)
        yield c


@pytest.fixture
def api(conn):
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id="AKIDTEST",
        aws_secret_access_key="secret",
        config=signing_config(),
    )
    app.dependency_overrides[get_db] = lambda: conn
    app.dependency_overrides[get_s3] = lambda: s3
    app.dependency_overrides[get_settings] = lambda: Settings(models_bucket=MODELS_BUCKET)
    yield handler
    app.dependency_overrides.clear()


def _crop(conn, code: str, display_name: str, description: str | None = None) -> int:
    return conn.execute(
        "INSERT INTO crops (code, display_name, description) VALUES (%s, %s, %s) RETURNING id",
        (code, display_name, description),
    ).fetchone()[0]


def _dataset(conn, name: str, crop_id: int, display_name: str) -> int:
    dataset_id = conn.execute(
        "INSERT INTO datasets (name, crop_id, display_name) VALUES (%s, %s, %s) RETURNING id",
        (name, crop_id, display_name),
    ).fetchone()[0]
    conn.execute("INSERT INTO dataset_versions (dataset_id, version) VALUES (%s, 0)", (dataset_id,))
    return dataset_id


def _model(conn, dataset_id: int, name: str, version: int, exported: bool = True) -> int:
    """A model version trained by its own (finished) round."""
    round_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, mode, strategy, k, seed,"
        " status, started_by, commit_sha)"
        " SELECT %s, %s, id, 'production', 'random', 5, 0, 'succeeded', 'duytq5', %s"
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


def _release(conn, model_version_id: int, action: str = "promote") -> None:
    conn.execute(
        "INSERT INTO model_releases (model_version_id, action, released_by, commit_sha)"
        " VALUES (%s, %s, 'duytq5', %s)",
        (model_version_id, action, uuid.uuid4().hex + "0" * 8),
    )


@pytest.fixture
def catalog(conn):
    """Coffee (RoCoLe, released twice), durian (leaf released, fruit not),
    and a crop with no dataset."""
    coffee = _crop(conn, "coffee", "Cà phê")
    durian = _crop(conn, "durian", "Sầu riêng", "Cây ăn quả")
    _crop(conn, "pepper", "Hồ tiêu")
    rocole = _dataset(conn, "rocole", coffee, "Lá cà phê (RoCoLe)")
    leaf = _dataset(conn, "durian", durian, "Lá sầu riêng")
    _dataset(conn, "durian_fruit", durian, "Quả sầu riêng")
    models = {
        "rocole/1": _model(conn, rocole, "rocole-fastvit", 1),
        "rocole/2": _model(conn, rocole, "rocole-fastvit", 2),
        "durian/1": _model(conn, leaf, "durian-fastvit", 1),
    }
    _release(conn, models["rocole/1"])
    _release(conn, models["rocole/2"])
    _release(conn, models["durian/1"])
    return models


# GET /v1/crops


def test_crops_empty(api):
    assert call(api, "/v1/crops")[:2] == (200, {"crops": []})


def test_crops_lists_released_datasets_only(api, catalog):
    status, body, _ = call(api, "/v1/crops")
    assert status == 200
    assert body == {
        "crops": [
            {
                "code": "coffee",
                "display_name": "Cà phê",
                "description": None,
                "datasets": [
                    {"name": "rocole", "display_name": "Lá cà phê (RoCoLe)", "version": 2}
                ],
            },
            {
                "code": "durian",
                "display_name": "Sầu riêng",
                "description": "Cây ăn quả",
                "datasets": [{"name": "durian", "display_name": "Lá sầu riêng", "version": 1}],
            },
        ]
    }


def test_crops_follow_a_rollback(api, conn, catalog):
    _release(conn, catalog["rocole/1"], "rollback")
    _, body, _ = call(api, "/v1/crops")
    assert body["crops"][0]["datasets"][0]["version"] == 1


def test_crops_list_every_released_dataset_of_a_crop(api, conn, catalog):
    fruit = conn.execute("SELECT id FROM datasets WHERE name = 'durian_fruit'").fetchone()[0]
    _release(conn, _model(conn, fruit, "durian-fruit-fastvit", 4))
    _, body, _ = call(api, "/v1/crops")
    durian = body["crops"][1]
    assert [d["name"] for d in durian["datasets"]] == ["durian", "durian_fruit"]
    assert durian["datasets"][1]["version"] == 4


# GET /v1/devices/config


def _presigned(url: str) -> tuple[str, str, dict[str, str]]:
    parts = urlsplit(url)
    return parts.netloc, parts.path, {k: v[0] for k, v in parse_qs(parts.query).items()}


def test_config_returns_the_current_model(api, catalog):
    status, body, _ = call(api, "/v1/devices/config", query={"dataset": "rocole"})
    assert status == 200
    assert {k: body[k] for k in ("dataset", "model", "version", "sha256", "expires_in")} == {
        "dataset": "rocole",
        "model": "rocole-fastvit",
        "version": 2,
        "sha256": SHA,
        "expires_in": 900,
    }
    for field, file in (("model_url", "model.onnx"), ("manifest_url", "manifest.json")):
        host, path, query = _presigned(body[field])
        assert host == f"{MODELS_BUCKET}.s3.{REGION}.amazonaws.com"
        assert path == f"/edge/rocole-fastvit/2/{file}"
        assert query["X-Amz-Algorithm"] == "AWS4-HMAC-SHA256"
        assert query["X-Amz-Expires"] == "900"


def test_config_follows_a_rollback(api, conn, catalog):
    _release(conn, catalog["rocole/1"], "rollback")
    _, body, _ = call(api, "/v1/devices/config", query={"dataset": "rocole"})
    assert body["version"] == 1
    assert "/edge/rocole-fastvit/1/model.onnx" in body["model_url"]


def test_config_needs_a_dataset(api):
    status, body, _ = call(api, "/v1/devices/config")
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "dataset" in body["error"]["message"]


def test_config_rejects_a_malformed_dataset_name(api):
    status, body, _ = call(api, "/v1/devices/config", query={"dataset": "../etc"})
    assert (status, body["error"]["code"]) == (400, "invalid_request")


@pytest.mark.parametrize("dataset", ["nope", "durian_fruit"])
def test_config_404_unknown_or_unreleased(api, catalog, dataset):
    status, body, _ = call(api, "/v1/devices/config", query={"dataset": dataset})
    assert status == 404
    assert body["error"]["code"] == "not_found"


def test_a_release_that_was_not_exported_is_not_served(api, conn, catalog):
    # Cannot happen through model-release (it records the export first); if it
    # did, the app keeps the model it has instead of getting a broken config.
    fruit = conn.execute("SELECT id FROM datasets WHERE name = 'durian_fruit'").fetchone()[0]
    _release(conn, _model(conn, fruit, "durian-fruit-fastvit", 1, exported=False))
    assert call(api, "/v1/devices/config", query={"dataset": "durian_fruit"})[0] == 404
    assert [c["code"] for c in call(api, "/v1/crops")[1]["crops"]] == ["coffee", "durian"]
    assert len(call(api, "/v1/crops")[1]["crops"][1]["datasets"]) == 1


# Postgres down (EC2 stopped for the night)


def test_database_unreachable_is_503(api):
    def down():
        raise psycopg.OperationalError("connection timeout expired")

    app.dependency_overrides[get_db] = down
    for path, query in (("/v1/crops", None), ("/v1/devices/config", {"dataset": "rocole"})):
        status, body, _ = call(api, path, query=query)
        assert status == 503
        assert body["error"]["code"] == "unavailable"


def test_db_connection_is_reused_until_it_drops(monkeypatch):
    class Conn:
        closed = False

    opened: list[Conn] = []
    monkeypatch.setattr(deps, "open_db", lambda: opened.append(Conn()) or opened[-1])
    monkeypatch.setattr(deps, "_conn", None)
    first = deps.get_db()
    assert deps.get_db() is first
    first.closed = True  # psycopg marks a dropped connection closed (and broken)
    assert deps.get_db() is opened[1]
    assert len(opened) == 2
