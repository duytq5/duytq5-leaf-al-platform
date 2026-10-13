"""The label step of the al-round state machine and the database side of the
Label Studio webhook (design doc F2). Both run inside the VPC: they need
Postgres, and Label Studio on the host's private IP.

  start_labeling  (AwaitLabels, waitForTaskToken) makes or finds the round's
                  Label Studio project, imports one task per selected image with
                  the model's prediction preselected, and stores the task token
  record_label    called by the webhook Lambda for each submitted annotation:
                  appends the expert's label (or marks the image rejected) and,
                  once all k images are done, hands back the task token

The webhook Lambda (lambdas/label_studio_hook.py) runs outside the VPC and
calls SendTaskSuccess with that token: these Lambdas have no route to the Step
Functions API (no NAT gateway).

Both are safe to run twice. A retried label step finds the project by the round
id in its description and imports tasks only into an empty project; labels are
append-only with one row per (image, round), so a repeated webhook stores
nothing new. Webhooks of one round are serialized on the round row, so the last
two labels arriving together cannot both miss that the round is complete.
"""

import logging
import os
from typing import Any
from uuid import UUID

import numpy as np
import psycopg

from common.s3keys import raw_key, scores_key
from lambdas import label_studio as ls
from lambdas.postgres import open_db
from lambdas.round import S3Store, Store, _classes, _running, _selection, read_scores

log = logging.getLogger(__name__)
RAW_PREFIX = "raw/"


def start_labeling(
    conn: psycopg.Connection,
    store: Store,
    studio: ls.LabelStudio,
    round_id: UUID,
    task_token: str,
    *,
    bucket: str,
    webhook_url: str,
    webhook_secret: str,
) -> dict[str, Any]:
    rnd = _running(conn, round_id)
    ids = _selection(store, round_id) or []
    # Train images only: test-split images never reach Label Studio.
    images = conn.execute(
        "SELECT id, sha256 FROM images WHERE id = ANY(%s) AND dataset_id = %s"
        " AND split = 'train' ORDER BY id",
        (ids, rnd.dataset_id),
    ).fetchall()
    classes = _classes(conn, rnd.dataset_id)

    description = ls.project_description(round_id)
    project_id = conn.execute(
        "SELECT ls_project_id FROM rounds WHERE id = %s", (round_id,)
    ).fetchone()[0]
    if project_id is None:
        found = [p["id"] for p in studio.projects() if p.get("description") == description]
        project_id = (
            found[0]
            if found
            else studio.create_project(
                ls.project_title(rnd.dataset, round_id), description, ls.label_config(classes)
            )
        )
    # Stored before the tasks exist, so no webhook can arrive for a round
    # without a token to resume.
    conn.execute(
        "UPDATE rounds SET ls_project_id = %s, label_task_token = %s WHERE id = %s",
        (project_id, task_token, round_id),
    )

    if not studio.s3_storages(project_id):
        studio.add_s3_storage(project_id, bucket, RAW_PREFIX)
    if not any(h.get("url") == webhook_url for h in studio.webhooks(project_id)):
        studio.add_webhook(project_id, webhook_url, {ls.SECRET_HEADER: webhook_secret})

    if studio.project(project_id).get("task_number", 0) == 0:
        predicted = _predictions(store, round_id, [c["code"] for c in classes])
        tasks = []
        for image_id, sha256 in images:
            task: dict[str, Any] = {
                "data": {
                    "image": f"s3://{bucket}/{raw_key(rnd.dataset, sha256)}",
                    "image_id": image_id,
                    "round_id": str(round_id),
                }
            }
            if image_id in predicted:
                code, score = predicted[image_id]
                task["predictions"] = [ls.prediction(code, score, round_id)]
            tasks.append(task)
        studio.import_tasks(project_id, tasks)
    return {"project_id": project_id, "tasks": len(images)}


def _predictions(store: Store, round_id: UUID, codes: list[str]) -> dict[int, tuple[str, float]]:
    """The score job's top class and its probability for each pool image."""
    body = store.get(scores_key(round_id))
    if body is None:
        return {}
    pool = read_scores(body)
    top = pool.probs.argmax(axis=1)
    best = np.take_along_axis(pool.probs, top[:, None], axis=1)[:, 0]
    return {
        int(i): (codes[int(t)], round(float(p), 4))
        for i, t, p in zip(pool.ids, top, best, strict=True)
    }


_IGNORED = {"status": "ignored"}


def record_label(conn: psycopg.Connection, store: Store, event: ls.LabelEvent) -> dict[str, Any]:
    """Append the expert's answer for one image of a round that waits for its
    labels. Returns the status, the progress (done of k), and once done the task
    token and the project to mark DONE."""
    ids = _selection(store, event.round_id)
    with conn.transaction():
        row = conn.execute(
            "SELECT r.status, r.label_task_token, r.dataset_id, r.ls_project_id, d.name"
            " FROM rounds r JOIN datasets d ON d.id = r.dataset_id WHERE r.id = %s"
            " FOR UPDATE OF r",
            (event.round_id,),
        ).fetchone()
        if row is None or ids is None or event.image_id not in ids:
            return _IGNORED
        status, token, dataset_id, project_id, dataset = row
        if status != "running" or token is None:
            return _IGNORED

        (image_status, labeled) = conn.execute(
            "SELECT status, EXISTS (SELECT 1 FROM labels WHERE image_id = i.id AND round_id = %s)"
            " FROM images i WHERE id = %s",
            (event.round_id, event.image_id),
        ).fetchone()
        # The first answer counts: a label after "not a leaf", or "not a leaf"
        # after a label, changes nothing.
        if event.choice == ls.NOT_LEAF:
            if labeled:
                return _IGNORED
            conn.execute(
                "UPDATE images SET status = 'rejected' WHERE id = %s AND status = 'queued'",
                (event.image_id,),
            )
            result = "rejected"
        else:
            found = conn.execute(
                "SELECT id FROM classes WHERE dataset_id = %s AND code = %s",
                (dataset_id, event.choice),
            ).fetchone()
            if found is None:
                return {"status": "unknown_code"}
            if image_status == "rejected":
                return _IGNORED
            conn.execute(
                "INSERT INTO labels (image_id, round_id, class_id, source, labeled_by)"
                " VALUES (%s, %s, %s, 'expert', %s) ON CONFLICT (image_id, round_id) DO NOTHING",
                (event.image_id, event.round_id, found[0], event.labeled_by),
            )
            result = "stored"

        (done,) = conn.execute(
            "SELECT count(*) FROM images i WHERE i.id = ANY(%s) AND (i.status = 'rejected'"
            " OR EXISTS (SELECT 1 FROM labels l WHERE l.image_id = i.id AND l.round_id = %s))",
            (ids, event.round_id),
        ).fetchone()
    out: dict[str, Any] = {"status": result, "done": done, "k": len(ids)}
    if done == len(ids):
        out |= {
            "task_token": token,
            "project_id": project_id,
            "title": ls.project_title(dataset, event.round_id, done=True),
        }
    return out


# --- Lambda handlers -------------------------------------------------------------


def _studio() -> ls.LabelStudio:
    return ls.LabelStudio(os.environ["LS_URL"], os.environ["LS_API_TOKEN"])


def start_handler(event: dict, _context) -> dict:
    bucket = os.environ["DATA_BUCKET"]
    with open_db() as conn:
        return start_labeling(
            conn,
            S3Store(bucket),
            _studio(),
            UUID(event["round_id"]),
            event["task_token"],
            bucket=bucket,
            webhook_url=os.environ["LS_WEBHOOK_URL"],
            webhook_secret=os.environ["LS_WEBHOOK_SECRET"],
        )


def record_handler(event: dict, _context) -> dict:
    """Invoked by the webhook Lambda with a parsed LabelEvent."""
    ev = ls.LabelEvent(
        round_id=UUID(event["round_id"]),
        image_id=int(event["image_id"]),
        choice=event["choice"],
        labeled_by=event["labeled_by"],
    )
    with open_db() as conn:
        out = record_label(conn, S3Store(os.environ["DATA_BUCKET"]), ev)
    if out.get("project_id") is not None:
        # Cosmetic: the round resumes whether or not the rename works.
        try:
            _studio().update_project(out["project_id"], title=out["title"])
        except ls.LabelStudioError:
            log.exception("could not mark Label Studio project %s DONE", out["project_id"])
    return out
