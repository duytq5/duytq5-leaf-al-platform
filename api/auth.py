"""Who is calling, from the claims API Gateway already verified.

The HTTP API's JWT authorizer checks the Cognito access token (signature,
issuer, client id, expiry) and rejects bad tokens with 401 before the Lambda
runs. The Lambda can only be invoked by API Gateway, so the claims in
`requestContext.authorizer.jwt.claims` are trusted here. Routes without the
authorizer (public ones) have no claims, and `current_user` answers 401.
"""

import json
from collections.abc import Callable

from fastapi import HTTPException, Request
from pydantic import BaseModel

GROUPS_CLAIM = "cognito:groups"


class User(BaseModel):
    sub: str
    username: str
    groups: list[str]


def parse_groups(raw) -> list[str]:
    """API Gateway passes array claims as "[a b]"; accept lists and JSON too."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(g) for g in raw]
    text = raw.strip()
    if text.startswith('["'):
        return [str(g) for g in json.loads(text)]
    return text.strip("[]").split()


def _claims(request: Request) -> dict | None:
    event = request.scope.get("aws.event") or {}
    authorizer = event.get("requestContext", {}).get("authorizer") or {}
    return (authorizer.get("jwt") or {}).get("claims")


def current_user(request: Request) -> User:
    claims = _claims(request)
    if not claims or not claims.get("sub"):
        raise HTTPException(401, "Sign in first.")
    return User(
        sub=claims["sub"],
        username=claims.get("username", claims["sub"]),
        groups=parse_groups(claims.get(GROUPS_CLAIM)),
    )


def require_groups(*allowed: str) -> Callable[[Request], User]:
    """FastAPI dependency: the caller must be in at least one of `allowed`."""
    if not allowed:
        raise ValueError("require_groups needs at least one group")

    def dependency(request: Request) -> User:
        user = current_user(request)
        if not set(user.groups) & set(allowed):
            raise HTTPException(403, f"Needs group: {' or '.join(allowed)}.")
        return user

    return dependency
