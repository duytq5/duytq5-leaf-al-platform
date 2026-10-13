"""Drive a Lambda handler with API Gateway HTTP API (payload 2.0) events.

`claims` is what the JWT authorizer passes after verifying a token; leave it
out for public routes and for requests without a token.
"""

import json
from urllib.parse import urlencode


def event(
    path: str,
    *,
    method: str = "GET",
    query: dict[str, str] | None = None,
    body: dict | None = None,
    claims: dict | None = None,
) -> dict:
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
        "routeKey": f"{method} {path}",
        "stage": "$default",
        "time": "13/Oct/2026:14:00:00 +0000",
        "timeEpoch": 1791900000000,
    }
    if claims is not None:
        context["authorizer"] = {"jwt": {"claims": claims, "scopes": None}}
    headers = {"host": "api.example.com"}
    if body is not None:
        headers["content-type"] = "application/json"
    out = {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "rawQueryString": urlencode(query or {}),
        "headers": headers,
        "requestContext": context,
        "isBase64Encoded": False,
    }
    if query:
        out["queryStringParameters"] = query
    if body is not None:
        out["body"] = json.dumps(body)
    return out


def call(handler, path: str, **kwargs) -> tuple[int, dict, dict]:
    """Status, JSON body and headers of one request."""
    response = handler(event(path, **kwargs), None)
    return response["statusCode"], json.loads(response["body"]), response.get("headers", {})
