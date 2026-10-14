"""Lambda steps of the model-release state machine (design doc F5).

  prepare  checks the release names a model version of the dataset and builds
           the export job from it, to run in the worker image of the commit
           that trained the model (the same recipe code)
  (export: SQS job the worker runs; it skips the export when the model is
           already in edge/, writes model.onnx then manifest.json, and moves
           the MLflow aliases)
  record   stores the export on model_versions and appends the model_releases
           row: `promote` the first time a version is released, `rollback`
           when it was released before

The app's current model is the dataset's latest model_releases row
(api/releases.py), so a rollback is a newer row pointing at an older version.

Both steps are safe to run twice: prepare only reads, and record adds no row
when the dataset's current release already is this version.
"""

from typing import Any

import psycopg

from common.jobs import ExportSpec
from common.release import ExportResult, ReleaseExecution, export_job_uuid
from common.s3keys import edge_prefix
from common.types import WORKER_IMAGE_REPO
from lambdas.logs import traced
from lambdas.postgres import open_db


class ReleaseError(RuntimeError):
    """The release cannot go on; the state machine fails and the workflow goes red."""


_VERSION = """
    SELECT mv.id, mv.labels, mv.onnx_sha256, r.commit_sha
      FROM model_versions mv
      JOIN rounds r ON r.id = mv.round_id
      JOIN datasets d ON d.id = r.dataset_id
     WHERE d.name = %s AND mv.mlflow_name = %s AND mv.mlflow_version = %s
"""

_CURRENT = """
    SELECT mr.model_version_id
      FROM model_releases mr
      JOIN model_versions mv ON mv.id = mr.model_version_id
      JOIN rounds r ON r.id = mv.round_id
     WHERE r.dataset_id = %s
     ORDER BY mr.released_at DESC, mr.id DESC
     LIMIT 1
"""


def _version(conn: psycopg.Connection, ev: ReleaseExecution) -> tuple[int, list, str | None, str]:
    row = conn.execute(_VERSION, (ev.dataset, ev.model, ev.version)).fetchone()
    if row is None:
        raise ReleaseError(
            f"{ev.model}/{ev.version} is not a model version of dataset {ev.dataset}"
        )
    return row


def prepare(conn: psycopg.Connection, ev: ReleaseExecution) -> dict[str, Any]:
    """The export job for the released version."""
    _, labels, _, trained_at = _version(conn, ev)
    spec = ExportSpec(
        model_uri=f"models:/{ev.model}/{ev.version}",
        output_prefix=edge_prefix(ev.model, ev.version),
        labels=labels,
    )
    return {
        "job_id": str(export_job_uuid(ev)),
        "image": f"{WORKER_IMAGE_REPO}:{trained_at}",
        "spec": spec.model_dump(mode="json"),
    }


def record(conn: psycopg.Connection, ev: ReleaseExecution, export: ExportResult) -> dict[str, Any]:
    """Store the export and make the version the dataset's current model."""
    with conn.transaction():
        # Releases of one dataset one at a time: the "current" check below
        # must not race another release.
        dataset = conn.execute(
            "SELECT id FROM datasets WHERE name = %s FOR UPDATE", (ev.dataset,)
        ).fetchone()
        if dataset is None:
            raise ReleaseError(f"no dataset {ev.dataset}")
        version_id, _, exported_sha, _ = _version(conn, ev)
        if exported_sha is None:
            conn.execute(
                "UPDATE model_versions SET onnx_s3_key = %s, onnx_sha256 = %s,"
                " size_bytes = %s, exported_at = now() WHERE id = %s",
                (export.onnx_s3_key, export.onnx_sha256, export.size_bytes, version_id),
            )
        elif exported_sha != export.onnx_sha256:
            # Installed apps verify the file against the recorded SHA-256.
            raise ReleaseError(
                f"{ev.model}/{ev.version} was exported before with SHA-256 {exported_sha},"
                f" the export job returned {export.onnx_sha256}"
            )

        current = conn.execute(_CURRENT, (dataset[0],)).fetchone()
        out = {"action": None, "model": ev.model, "version": ev.version}
        if current is not None and current[0] == version_id:
            return out  # already the current model: a retry, or the file changed otherwise
        released_before = conn.execute(
            "SELECT 1 FROM model_releases WHERE model_version_id = %s", (version_id,)
        ).fetchone()
        action = "rollback" if released_before else "promote"
        # A re-run of an older release commit (another release came after it)
        # finds its row already there and adds nothing.
        inserted = conn.execute(
            "INSERT INTO model_releases (model_version_id, action, released_by, commit_sha)"
            " VALUES (%s, %s, %s, %s) ON CONFLICT (model_version_id, commit_sha) DO NOTHING"
            " RETURNING id",
            (version_id, action, ev.released_by, ev.commit_sha),
        ).fetchone()
    return {**out, "action": action if inserted else None}


# --- Lambda handlers -------------------------------------------------------------


class _RecordEvent(ReleaseExecution):
    export: ExportResult


@traced
def prepare_handler(event: dict, _context) -> dict:
    ev = ReleaseExecution.model_validate(event)
    with open_db() as conn:
        return prepare(conn, ev)


@traced
def record_handler(event: dict, _context) -> dict:
    full = _RecordEvent.model_validate(event)
    ev = ReleaseExecution.model_validate(full.model_dump(exclude={"export"}))
    with open_db() as conn:
        return record(conn, ev, full.export)
