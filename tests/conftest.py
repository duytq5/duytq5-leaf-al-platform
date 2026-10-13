import json
import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

# CDK template tests don't build Lambda bundles (that downloads wheels); the
# bundle itself is tested in tests/test_api_infra.py.
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
