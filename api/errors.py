"""Error bodies: {"error": {"code": ..., "message": ...}} (design doc, Edge API).

Throttling 429s from API Gateway itself don't use this shape.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

ERROR_CODES = {
    400: "invalid_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "invalid_request",
    409: "conflict",
    429: "quota_exceeded",
}


def error_response(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    code = ERROR_CODES.get(status, "internal_error")
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status, headers=headers
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        if exc.status_code == 404 and message == "Not Found":
            message = "No such route."
        return error_response(exc.status_code, message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()))
        return error_response(400, f"{where}: {first.get('msg', 'invalid')}".strip(": "))
