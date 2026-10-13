"""Public app routes: GET /v1/crops and GET /v1/devices/config.

No token: API Gateway passes no claims. The current model of a dataset is its
latest model_releases row (design doc F5), against a real Postgres
(TEST_DATABASE_URL). Download links are presigned locally by botocore with
fake credentials, so no AWS call is made.
"""

import psycopg
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mangum")

from api import deps
from api.app import app
from api.deps import get_db
from tests.apidb import MODELS_BUCKET, REGION, SHA, make_catalog, model, presigned, release
from tests.apigw import call


@pytest.fixture
def catalog(db):
    return make_catalog(db)


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


def test_crops_follow_a_rollback(api, db, catalog):
    release(db, catalog["rocole/1"], "rollback")
    _, body, _ = call(api, "/v1/crops")
    assert body["crops"][0]["datasets"][0]["version"] == 1


def test_crops_list_every_released_dataset_of_a_crop(api, db, catalog):
    fruit = db.execute("SELECT id FROM datasets WHERE name = 'durian_fruit'").fetchone()[0]
    release(db, model(db, fruit, "durian-fruit-fastvit", 4))
    _, body, _ = call(api, "/v1/crops")
    durian = body["crops"][1]
    assert [d["name"] for d in durian["datasets"]] == ["durian", "durian_fruit"]
    assert durian["datasets"][1]["version"] == 4


# GET /v1/devices/config


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
        host, path, query = presigned(body[field])
        assert host == f"{MODELS_BUCKET}.s3.{REGION}.amazonaws.com"
        assert path == f"/edge/rocole-fastvit/2/{file}"
        assert query["X-Amz-Algorithm"] == "AWS4-HMAC-SHA256"
        assert query["X-Amz-Expires"] == "900"


def test_config_follows_a_rollback(api, db, catalog):
    release(db, catalog["rocole/1"], "rollback")
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


def test_a_release_that_was_not_exported_is_not_served(api, db, catalog):
    # Cannot happen through model-release (it records the export first); if it
    # did, the app keeps the model it has instead of getting a broken config.
    fruit = db.execute("SELECT id FROM datasets WHERE name = 'durian_fruit'").fetchone()[0]
    release(db, model(db, fruit, "durian-fruit-fastvit", 1, exported=False))
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
