"""The Label Studio pieces that need no database: the labeling config, project
names, the webhook body and the REST client (with a fake transport)."""

import io
import json
import urllib.error
import uuid
import xml.etree.ElementTree as ET

import pytest

from lambdas import label_studio as ls

CLASSES = [
    {"code": "coffee___healthy", "display_name": "Lá khỏe"},
    {"code": "coffee___rust", "display_name": "Gỉ sắt & <đốm>"},
]
ROUND = uuid.UUID("1b2c3d4e-0000-4000-8000-000000000000")


def test_label_config_stores_class_codes_and_offers_not_a_leaf():
    root = ET.fromstring(ls.label_config(CLASSES))
    (image,) = root.iter("Image")
    assert image.get("name") == ls.OBJECT
    assert image.get("value") == "$image"
    (choices,) = root.iter("Choices")
    assert choices.get("name") == ls.CONTROL
    assert choices.get("toName") == ls.OBJECT
    assert choices.get("choice") == "single-radio"
    assert choices.get("required") == "true"
    # Results carry the alias: the class code, never the display name.
    assert [(c.get("alias"), c.get("value")) for c in choices.iter("Choice")] == [
        ("coffee___healthy", "Lá khỏe"),
        ("coffee___rust", "Gỉ sắt & <đốm>"),
        (ls.NOT_LEAF, "Không phải lá"),
    ]


def test_label_config_refuses_a_class_named_like_not_a_leaf():
    with pytest.raises(ValueError, match=ls.NOT_LEAF):
        ls.label_config([*CLASSES, {"code": ls.NOT_LEAF, "display_name": "x"}])


def test_project_title_fits_label_studio_and_shows_the_state():
    active = ls.project_title("a-very-long-dataset-name-for-durian-leaves", ROUND)
    done = ls.project_title("rocole", ROUND, done=True)
    assert active.startswith("ACTIVE ")
    assert done == "DONE rocole 1b2c3d4e"
    assert len(active) <= ls.TITLE_MAX
    assert ls.project_description(ROUND) == f"al-round {ROUND}"


def test_prediction_preselects_the_class():
    assert ls.prediction("coffee___rust", 0.8125, ROUND) == {
        "model_version": f"round {ROUND}",
        "score": 0.8125,
        "result": [
            {
                "from_name": ls.CONTROL,
                "to_name": ls.OBJECT,
                "type": "choices",
                "value": {"choices": ["coffee___rust"]},
            }
        ],
    }


def _hook(choices=("coffee___rust",), **overrides) -> dict:
    body = {
        "action": "ANNOTATION_CREATED",
        "annotation": {
            "id": 9,
            "completed_by": 3,
            "was_cancelled": False,
            "result": [
                {"from_name": ls.CONTROL, "to_name": ls.OBJECT, "type": "choices",
                 "value": {"choices": list(choices)}},
            ],
        },
        "task": {"id": 5, "data": {"image": "s3://b/raw/x.jpg", "image_id": 42,
                                   "round_id": str(ROUND)}},
        "project": {"id": 1},
    }  # fmt: skip
    body.update(overrides)
    return body


def test_parse_webhook_reads_only_what_it_needs():
    assert ls.parse_webhook(_hook()) == ls.LabelEvent(
        round_id=ROUND, image_id=42, choice="coffee___rust", labeled_by="label-studio:3"
    )


def test_parse_webhook_takes_the_email_when_the_user_is_expanded():
    body = _hook()
    body["annotation"]["completed_by"] = {"id": 3, "email": "expert@example.com"}
    assert ls.parse_webhook(body).labeled_by == "expert@example.com"


@pytest.mark.parametrize(
    "body",
    [
        _hook(action="ANNOTATION_UPDATED"),  # edits after submit are not recorded
        _hook(action="PROJECT_CREATED"),
        _hook(task={"id": 5, "data": {"image": "s3://b/x.jpg"}}),  # not a round's task
        {"action": "ANNOTATION_CREATED"},
    ],
)
def test_parse_webhook_ignores_other_events(body):
    assert ls.parse_webhook(body) is None


def test_parse_webhook_ignores_skipped_tasks():
    body = _hook(choices=())
    body["annotation"]["was_cancelled"] = True
    assert ls.parse_webhook(body) is None


def test_parse_webhook_refuses_a_round_task_without_a_choice():
    with pytest.raises(ValueError):
        ls.parse_webhook(_hook(choices=()))
    body = _hook()
    body["task"]["data"]["image_id"] = "not-a-number"
    with pytest.raises(ValueError):
        ls.parse_webhook(body)


class FakeOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return io.BytesIO(json.dumps(out).encode())


def test_client_sends_the_token_and_json():
    opener = FakeOpener([{"id": 7}])
    client = ls.LabelStudio("http://10.0.1.5:8080/", "tok", opener=opener)
    assert client.create_project("ACTIVE toy 1b2c3d4e", "al-round x", "<View/>") == 7
    (req,) = opener.requests
    assert req.full_url == "http://10.0.1.5:8080/api/projects/"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Token tok"
    assert req.get_header("Content-type") == "application/json"
    assert json.loads(req.data) == {
        "title": "ACTIVE toy 1b2c3d4e",
        "description": "al-round x",
        "label_config": "<View/>",
        "maximum_annotations": 1,
        "show_skip_button": False,
    }


def test_s3_storage_signs_for_the_bucket_region(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    opener = FakeOpener([{"id": 3}])
    client = ls.LabelStudio("http://10.0.1.5:8080/", "tok", opener=opener)
    client.add_s3_storage(7, "data-bucket", "raw/")
    (req,) = opener.requests
    assert req.full_url == "http://10.0.1.5:8080/api/storages/s3/"
    body = json.loads(req.data)
    # Without it, Label Studio signs for us-east-1 and S3 refuses the image.
    assert body["region_name"] == "ap-southeast-2"
    assert body["presign"] is True
    assert (body["project"], body["bucket"], body["prefix"]) == (7, "data-bucket", "raw/")


def test_client_reads_paginated_and_plain_lists():
    opener = FakeOpener([{"count": 1, "results": [{"id": 1}]}, [{"id": 2}]])
    client = ls.LabelStudio("http://ls:8080", "tok", opener=opener)
    assert client.projects() == [{"id": 1}]
    assert client.webhooks(1) == [{"id": 2}]
    assert opener.requests[1].full_url == "http://ls:8080/api/webhooks/?project=1"


def test_client_errors_name_the_call():
    http = urllib.error.HTTPError("http://ls/api/projects/", 400, "Bad", {}, io.BytesIO(b"no"))
    client = ls.LabelStudio("http://ls", "tok", opener=FakeOpener([http]))
    with pytest.raises(ls.LabelStudioError, match="POST /api/projects/1/import: HTTP 400"):
        client.import_tasks(1, [])
    client = ls.LabelStudio("http://ls", "tok", opener=FakeOpener([urllib.error.URLError("x")]))
    with pytest.raises(ls.LabelStudioError, match="unreachable"):
        client.projects()
