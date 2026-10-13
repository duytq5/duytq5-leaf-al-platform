"""POST /v1/hooks/label-studio: Label Studio's annotation webhook (design doc F2).

Runs outside the VPC, behind the HTTP API with no JWT: Label Studio proves
itself with the shared secret header instead. The Postgres side runs in the
VPC (lambdas.labeling.record_handler, invoked synchronously); this Lambda
cannot reach Postgres, and that one cannot reach the Step Functions API, so
the work is split between them:

  1. reject a missing or wrong X-Webhook-Secret (401)
  2. keep only a round task's submitted annotation (anything else: 200 ignored)
  3. record it in the VPC Lambda (400 for a class code the dataset lacks)
  4. once all k images are done, SendTaskSuccess with the round's token

A task token works once. The same webhook twice, or two last labels at once,
end in InvalidToken or TaskTimedOut on the second call, which counts as
already resumed.

Responses: 200 {"status": "stored" | "rejected" | "ignored", ...},
401 / 400 {"error": {"code", "message"}}.
"""

import base64
import hmac
import json
import os
from functools import cache

import boto3
from botocore.exceptions import ClientError

from lambdas import label_studio as ls

ALREADY_RESUMED = {"InvalidToken", "TaskTimedOut", "TaskDoesNotExist"}


@cache
def _lambda():
    return boto3.client("lambda")


@cache
def _sfn():
    return boto3.client("stepfunctions")


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error(status: int, code: str, message: str) -> dict:
    return _response(status, {"error": {"code": code, "message": message}})


def handler(event: dict, _context) -> dict:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    given = headers.get(ls.SECRET_HEADER.lower(), "")
    if not hmac.compare_digest(given.encode(), os.environ["LS_WEBHOOK_SECRET"].encode()):
        return _error(401, "unauthenticated", f"missing or wrong {ls.SECRET_HEADER}")

    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    try:
        label = ls.parse_webhook(json.loads(raw))
    except (ValueError, AttributeError) as e:
        return _error(400, "invalid_request", str(e))
    if label is None:
        return _response(200, {"status": "ignored"})

    resp = _lambda().invoke(
        FunctionName=os.environ["RECORD_LABEL_FUNCTION"],
        Payload=json.dumps(
            {
                "round_id": str(label.round_id),
                "image_id": label.image_id,
                "choice": label.choice,
                "labeled_by": label.labeled_by,
            }
        ),
    )
    result = json.loads(resp["Payload"].read())
    if resp.get("FunctionError"):
        raise RuntimeError(f"record_label failed: {result.get('errorMessage', result)}")
    if result["status"] == "unknown_code":
        return _error(400, "invalid_request", f"unknown class code {label.choice!r}")

    token = result.pop("task_token", None)
    result.pop("project_id", None)
    result.pop("title", None)
    if token is not None:
        try:
            _sfn().send_task_success(
                taskToken=token, output=json.dumps({"labeled": result["done"]})
            )
            result["resumed"] = True
        except ClientError as e:
            if e.response["Error"]["Code"] not in ALREADY_RESUMED:
                raise
            result["resumed"] = False
    return _response(200, result)
