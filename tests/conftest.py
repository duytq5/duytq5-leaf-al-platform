import json
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

# CDK template tests don't build Lambda bundles (that downloads Linux wheels);
# cdk.App reads this variable. The API bundle is tested in tests/test_api_infra.py.
os.environ.setdefault("CDK_CONTEXT_JSON", json.dumps({"aws:cdk:bundling-stacks": []}))


@pytest.fixture
def pg_url():
    """A fresh, empty database for one test, dropped afterwards.

    Needs TEST_DATABASE_URL pointing at a Postgres where the user may create
    databases (CI runs a postgres:16 service). Skipped when it is not set.
    """
    base = os.environ.get("TEST_DATABASE_URL")
    if not base:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg

    name = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    parts = urlsplit(base)
    yield urlunsplit(parts._replace(path=f"/{name}"))
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


@pytest.fixture
def db(pg_url):
    """A migrated database for the API tests."""
    from db import connect
    from db.migrate import migrate

    with connect(pg_url) as conn:
        migrate(conn)
        yield conn


@pytest.fixture
def api(db):
    """The API Lambda handler on the test database, with S3 links signed by
    fake credentials (signing makes no AWS call)."""
    pytest.importorskip("fastapi")
    from api.app import app, handler
    from api.deps import get_db, get_s3, get_settings
    from tests.apidb import SETTINGS, fake_s3

    s3 = fake_s3()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_s3] = lambda: s3
    app.dependency_overrides[get_settings] = lambda: SETTINGS
    yield handler
    app.dependency_overrides.clear()
