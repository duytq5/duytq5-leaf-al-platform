"""Label Studio for the label step (design doc F2): the labeling config, project
names, the webhook body and a small REST client.

One project per round, named "ACTIVE <dataset> <round id prefix>" while it
waits for labels and "DONE ..." once all k images are labeled; its description
holds the full round id, which is how a retried step finds it again. After the
round merges, its tasks and annotations are exported to S3 and the project is
deleted (lambdas/labeling.py close_project).

Tasks carry s3:// links to raw/. The project's S3 storage (no sync, presign on)
lets Label Studio sign them with the EC2 instance role each time a labeler
opens a task, so no link in a task can expire while the round waits.

Results store the class code (the Choice alias), never the display name, so a
renamed class does not change what the webhook receives. "Không phải lá" (not a
leaf) stores NOT_LEAF.

Standard library only: the webhook Lambda outside the VPC imports this module
too, and it should start fast.
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from uuid import UUID
from xml.sax.saxutils import quoteattr

CONTROL = "label"
OBJECT = "image"
NOT_LEAF = "not_leaf"
NOT_LEAF_NAME = "Không phải lá"
SECRET_HEADER = "X-Webhook-Secret"
WEBHOOK_ACTION = "ANNOTATION_CREATED"
# Label Studio's PROJECT_TITLE_MAX_LEN.
TITLE_MAX = 50
TIMEOUT_SECONDS = 20


class LabelStudioError(RuntimeError):
    """Label Studio answered with an error or could not be reached. The state
    machine retries the label step on it (Label Studio may be restarting)."""


class LabelStudioNotFound(LabelStudioError):
    """Label Studio answered 404: the project or user does not exist."""


def label_config(classes: list[dict[str, str]]) -> str:
    """One image, one required single choice: the dataset's classes in output
    order, then "not a leaf"."""
    codes = [c["code"] for c in classes]
    if NOT_LEAF in codes:
        raise ValueError(f"class code {NOT_LEAF!r} is reserved for 'not a leaf'")
    choices = [(c["display_name"], c["code"]) for c in classes] + [(NOT_LEAF_NAME, NOT_LEAF)]
    lines = [
        "<View>",
        f'  <Image name="{OBJECT}" value="$image" zoom="true"/>',
        f'  <Choices name="{CONTROL}" toName="{OBJECT}" choice="single-radio" required="true">',
        *(
            f"    <Choice value={quoteattr(name)} alias={quoteattr(code)}/>"
            for name, code in choices
        ),
        "  </Choices>",
        "</View>",
    ]
    return "\n".join(lines)


def project_title(dataset: str, round_id: UUID, *, done: bool = False) -> str:
    state = "DONE" if done else "ACTIVE"
    short = str(round_id)[:8]
    room = TITLE_MAX - len(state) - len(short) - 2
    return f"{state} {dataset[:room]} {short}"


def project_description(round_id: UUID) -> str:
    return f"al-round {round_id}"


def prediction(code: str, score: float, round_id: UUID) -> dict[str, Any]:
    """The pre-annotation: the class the round's model predicted, selected."""
    return {
        "model_version": f"round {round_id}",
        "score": score,
        "result": [
            {
                "from_name": CONTROL,
                "to_name": OBJECT,
                "type": "choices",
                "value": {"choices": [code]},
            }
        ],
    }


@dataclass(frozen=True)
class LabelEvent:
    round_id: UUID
    image_id: int
    choice: str  # a class code or NOT_LEAF
    labeled_by: str


def parse_webhook(body: dict[str, Any]) -> LabelEvent | None:
    """The fields of an annotation webhook the round needs. None for anything
    else (other actions, skipped tasks, tasks that are not a round's); edits
    after submit are not recorded. ValueError for a round's task whose
    annotation has no usable choice."""
    if body.get("action") != WEBHOOK_ACTION:
        return None
    annotation = body.get("annotation") or {}
    data = (body.get("task") or {}).get("data") or {}
    if annotation.get("was_cancelled") or "round_id" not in data or "image_id" not in data:
        return None
    choices = [
        r.get("value", {}).get("choices") or []
        for r in annotation.get("result") or []
        if r.get("type") == "choices" and r.get("from_name") == CONTROL
    ]
    if not choices or not choices[0] or not isinstance(choices[0][0], str) or not choices[0][0]:
        raise ValueError("the annotation has no choice")
    who = annotation.get("completed_by")
    if isinstance(who, dict):
        who = who.get("email") or who.get("id")
    labeled_by = who if isinstance(who, str) else f"label-studio:{who}"
    try:
        return LabelEvent(
            round_id=UUID(str(data["round_id"])),
            image_id=int(data["image_id"]),
            choice=choices[0][0],
            labeled_by=labeled_by,
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"bad task data: {e}") from e


class LabelStudio:
    """The few REST calls the label step makes, with the admin's API token.
    The Lambdas call Label Studio on the host's private IP, inside the VPC."""

    def __init__(self, url: str, token: str, opener=urllib.request.urlopen) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.opener = opener

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        request = urllib.request.Request(
            self.url + path,
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Token {self.token}", "Content-Type": "application/json"},
        )
        try:
            with self.opener(request, timeout=TIMEOUT_SECONDS) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            error = LabelStudioNotFound if e.code == 404 else LabelStudioError
            raise error(f"{method} {path}: HTTP {e.code} {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LabelStudioError(f"{method} {path}: Label Studio unreachable ({e})") from e
        return json.loads(raw) if raw else None

    @staticmethod
    def _list(out: Any) -> list[dict]:
        return out["results"] if isinstance(out, dict) else out

    def projects(self) -> list[dict]:
        return self._list(self._call("GET", "/api/projects/?page_size=1000"))

    def project(self, project_id: int) -> dict:
        return self._call("GET", f"/api/projects/{project_id}/")

    def create_project(self, title: str, description: str, label_config: str) -> int:
        body = {
            "title": title,
            "description": description,
            "label_config": label_config,
            # One expert answer per image; skipping would leave the round waiting.
            "maximum_annotations": 1,
            "show_skip_button": False,
        }
        return self._call("POST", "/api/projects/", body)["id"]

    def update_project(self, project_id: int, **fields: Any) -> None:
        self._call("PATCH", f"/api/projects/{project_id}/", fields)

    def s3_storages(self, project_id: int) -> list[dict]:
        return self._list(self._call("GET", f"/api/storages/s3/?project={project_id}"))

    def add_s3_storage(self, project_id: int, bucket: str, prefix: str) -> None:
        # No keys: Label Studio uses the instance role. Never synced; it only
        # signs the s3:// links of imported tasks. Without region_name it signs
        # for us-east-1, and S3 refuses the links of a bucket in another
        # region; the bucket is in this Lambda's region.
        body = {
            "project": project_id,
            "title": "raw images",
            "bucket": bucket,
            "prefix": prefix,
            "region_name": os.environ["AWS_REGION"],
            "presign": True,
            "presign_ttl": 15,
            "use_blob_urls": False,
        }
        self._call("POST", "/api/storages/s3/", body)

    def webhooks(self, project_id: int) -> list[dict]:
        return self._list(self._call("GET", f"/api/webhooks/?project={project_id}"))

    def add_webhook(self, project_id: int, url: str, headers: dict[str, str]) -> None:
        body = {
            "project": project_id,
            "url": url,
            "headers": headers,
            "send_payload": True,
            "send_for_all_actions": False,
            "actions": [WEBHOOK_ACTION],
            "is_active": True,
        }
        self._call("POST", "/api/webhooks/", body)

    def import_tasks(self, project_id: int, tasks: list[dict]) -> None:
        self._call("POST", f"/api/projects/{project_id}/import", tasks)

    def export(self, project_id: int) -> list[dict]:
        """Every task of the project with its data, annotations and predictions."""
        return self._call(
            "GET", f"/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true"
        )

    def delete_project(self, project_id: int) -> bool:
        """Delete the project with its tasks, storages and webhooks. False when it
        was already gone."""
        try:
            self._call("DELETE", f"/api/projects/{project_id}/")
        except LabelStudioNotFound:
            return False
        return True

    def user(self, user_id: int) -> dict:
        return self._call("GET", f"/api/users/{user_id}/")
