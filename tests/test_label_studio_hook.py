"""The webhook Lambda outside the VPC: checks the secret, hands the label to the
VPC Lambda and resumes the round. AWS clients are fakes."""

import base64
import io
import json
import uuid

import pytest
from botocore.exceptions import ClientError

from lambdas import label_studio as ls
from lambdas import label_studio_hook as hook

SECRET = "s" * 40
ROUND = uuid.uuid4()


class FakeLambda:
    def __init__(self, result: dict | None = None, error: bool = False) -> None:
        self.result = result or {"status": "stored", "done": 1, "k": 3}
        self.error = error
        self.calls: list[dict] = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        out = {"StatusCode": 200, "Payload": io.BytesIO(json.dumps(self.result).encode())}
        if self.error:
            out["FunctionError"] = "Unhandled"
        return out


class FakeSfn:
    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []

    def send_task_success(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise ClientError({"Error": {"Code": self.error, "Message": "x"}}, "SendTaskSuccess")


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.setenv("LS_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("RECORD_LABEL_FUNCTION", "record-fn")

    def install(lam: FakeLambda, sfn: FakeSfn):
        monkeypatch.setattr(hook, "_lambda", lambda: lam)
        monkeypatch.setattr(hook, "_sfn", lambda: sfn)
        return lam, sfn

    return install


def _body(choice="a", action="ANNOTATION_CREATED") -> dict:
    return {
        "action": action,
        "annotation": {
            "completed_by": 2,
            "result": [
                {"from_name": ls.CONTROL, "type": "choices", "value": {"choices": [choice]}}
            ],
        },
        "task": {"data": {"image_id": 7, "round_id": str(ROUND)}},
    }


def _request(body, secret: str | None = SECRET, b64: bool = False) -> dict:
    raw = json.dumps(body)
    headers = {"content-type": "application/json"}
    if secret is not None:
        headers["x-webhook-secret"] = secret
    return {
        "headers": headers,
        "body": base64.b64encode(raw.encode()).decode() if b64 else raw,
        "isBase64Encoded": b64,
    }


def _call(event) -> tuple[int, dict]:
    out = hook.handler(event, None)
    return out["statusCode"], json.loads(out["body"])


@pytest.mark.parametrize("secret", [None, "", "wrong", SECRET + "x"])
def test_wrong_or_missing_secret_is_401(aws, secret):
    lam, sfn = aws(FakeLambda(), FakeSfn())
    status, body = _call(_request(_body(), secret=secret))
    assert status == 401
    assert body["error"]["code"] == "unauthenticated"
    assert lam.calls == sfn.calls == []


def test_a_label_goes_to_the_vpc_lambda(aws):
    lam, sfn = aws(FakeLambda(), FakeSfn())
    status, body = _call(_request(_body(), b64=True))
    assert (status, body) == (200, {"status": "stored", "done": 1, "k": 3})
    (call,) = lam.calls
    assert call["FunctionName"] == "record-fn"
    assert json.loads(call["Payload"]) == {
        "round_id": str(ROUND),
        "image_id": 7,
        "choice": "a",
        "labeled_by": "label-studio:2",
    }
    assert sfn.calls == []


def test_the_last_label_resumes_the_round(aws):
    result = {"status": "stored", "done": 3, "k": 3, "task_token": "tok", "project_id": 1,
              "title": "DONE toy 1b2c3d4e"}  # fmt: skip
    _, sfn = aws(FakeLambda(result), FakeSfn())
    status, body = _call(_request(_body()))
    assert status == 200
    assert body == {"status": "stored", "done": 3, "k": 3, "resumed": True}  # no token out
    assert sfn.calls == [{"taskToken": "tok", "output": json.dumps({"labeled": 3})}]


@pytest.mark.parametrize("code", ["InvalidToken", "TaskTimedOut", "TaskDoesNotExist"])
def test_a_token_used_before_counts_as_resumed(aws, code):
    result = {"status": "stored", "done": 3, "k": 3, "task_token": "tok"}
    aws(FakeLambda(result), FakeSfn(error=code))
    status, body = _call(_request(_body()))
    assert status == 200
    assert body["resumed"] is False


def test_other_step_functions_errors_fail_loudly(aws):
    aws(
        FakeLambda({"status": "stored", "done": 3, "k": 3, "task_token": "t"}),
        FakeSfn("Throttling"),
    )
    with pytest.raises(ClientError):
        hook.handler(_request(_body()), None)


def test_unknown_code_is_400(aws):
    aws(FakeLambda({"status": "unknown_code"}), FakeSfn())
    status, body = _call(_request(_body("zzz")))
    assert status == 400
    assert body["error"]["code"] == "invalid_request"


def test_other_events_are_ignored_without_touching_the_database(aws):
    lam, _ = aws(FakeLambda(), FakeSfn())
    assert _call(_request(_body(action="ANNOTATION_UPDATED"))) == (200, {"status": "ignored"})
    assert lam.calls == []


def test_bad_bodies_are_400(aws):
    lam, _ = aws(FakeLambda(), FakeSfn())
    status, _ = _call({"headers": {"x-webhook-secret": SECRET}, "body": "{not json"})
    assert status == 400
    status, _ = _call(_request(_body(choice="")))
    assert status == 400
    assert lam.calls == []


def test_a_failing_vpc_lambda_is_a_500(aws):
    aws(FakeLambda({"errorMessage": "boom"}, error=True), FakeSfn())
    with pytest.raises(RuntimeError, match="boom"):
        hook.handler(_request(_body()), None)
