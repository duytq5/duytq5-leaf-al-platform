"""The API trusts API Gateway for the token and checks groups itself.

API Gateway's JWT authorizer verifies the Cognito access token (signature,
issuer, client id, expiry) before the Lambda runs and passes the claims in
`requestContext.authorizer.jwt.claims`. These tests drive the real Lambda
handler with such events.
"""

import json
from typing import Annotated

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mangum")

from fastapi import Depends, FastAPI
from mangum import Mangum

from api.app import handler
from api.auth import User, parse_groups, require_groups
from api.errors import install_error_handlers


def _event(path: str, claims: dict | None = None, method: str = "GET") -> dict:
    context = {
        "accountId": "123456789012",
        "apiId": "api",
        "domainName": "api.example.com",
        "http": {
            "method": method,
            "path": path,
            "protocol": "HTTP/1.1",
            "sourceIp": "203.0.113.1",
            "userAgent": "pytest",
        },
        "requestId": "req",
        "routeKey": "ANY /v1/{proxy+}",
        "stage": "$default",
        "time": "13/Oct/2026:14:00:00 +0000",
        "timeEpoch": 1791900000000,
    }
    if claims is not None:
        context["authorizer"] = {"jwt": {"claims": claims, "scopes": None}}
    return {
        "version": "2.0",
        "routeKey": "ANY /v1/{proxy+}",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"host": "api.example.com"},
        "requestContext": context,
        "isBase64Encoded": False,
    }


def _call(app_handler, path: str, claims: dict | None = None) -> tuple[int, dict]:
    response = app_handler(_event(path, claims), None)
    return response["statusCode"], json.loads(response["body"])


# Claims as API Gateway passes them: arrays arrive as "[a b]" strings.
ADMIN = {"sub": "u-1", "username": "u-1", "cognito:groups": "[admins labelers]"}
UPLOADER = {"sub": "u-2", "username": "google_123", "cognito:groups": "[uploaders]"}
NO_GROUPS = {"sub": "u-3", "username": "u-3"}


@pytest.mark.parametrize(
    ("raw", "groups"),
    [
        (None, []),
        ("", []),
        ("[]", []),
        ("admins", ["admins"]),
        ("[admins labelers]", ["admins", "labelers"]),
        ('["admins","labelers"]', ["admins", "labelers"]),
        (["admins", "labelers"], ["admins", "labelers"]),
    ],
)
def test_parse_groups(raw, groups):
    assert parse_groups(raw) == groups


def test_me_without_claims_is_401():
    status, body = _call(handler, "/v1/me")
    assert status == 401
    assert body == {"error": {"code": "unauthenticated", "message": "Sign in first."}}


def test_me_returns_the_caller():
    status, body = _call(handler, "/v1/me", ADMIN)
    assert status == 200
    assert body == {"sub": "u-1", "username": "u-1", "groups": ["admins", "labelers"]}


def test_me_works_without_groups():
    status, body = _call(handler, "/v1/me", NO_GROUPS)
    assert status == 200
    assert body["groups"] == []


def test_unknown_route_uses_error_shape():
    status, body = _call(handler, "/v1/nope", ADMIN)
    assert status == 404
    assert body["error"]["code"] == "not_found"


@pytest.fixture(scope="module")
def admin_only():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/v1/admin-thing")
    def thing(user: Annotated[User, Depends(require_groups("admins"))]) -> dict:
        return {"sub": user.sub}

    @app.get("/v1/labeler-thing")
    def labeler_thing(
        user: Annotated[User, Depends(require_groups("labelers", "admins"))],
    ) -> dict:
        return {"sub": user.sub}

    return Mangum(app, lifespan="off")


def test_require_groups_401_without_claims(admin_only):
    status, body = _call(admin_only, "/v1/admin-thing")
    assert status == 401
    assert body["error"]["code"] == "unauthenticated"


def test_require_groups_403_wrong_group(admin_only):
    status, body = _call(admin_only, "/v1/admin-thing", UPLOADER)
    assert status == 403
    assert body == {"error": {"code": "forbidden", "message": "Needs group: admins."}}


def test_require_groups_any_of(admin_only):
    assert _call(admin_only, "/v1/admin-thing", ADMIN) == (200, {"sub": "u-1"})
    labeler = {"sub": "u-4", "cognito:groups": "[labelers]"}
    assert _call(admin_only, "/v1/labeler-thing", labeler) == (200, {"sub": "u-4"})
    assert _call(admin_only, "/v1/labeler-thing", UPLOADER)[0] == 403


def test_require_groups_needs_a_group():
    with pytest.raises(ValueError):
        require_groups()
