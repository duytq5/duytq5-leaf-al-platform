"""Round progress for labelers and admins.

The API Lambda has no route to Step Functions (no NAT), so the phase comes
from Postgres and the round's selection file, which the select step writes
before the label step starts:

  pending                                   starting
  running, no selections/{round_id}.json    training (snapshot, train, score, select)
  running, some selected images not done    awaiting_labels
  running, every selected image done        merging
  succeeded, failed, cancelled              None

An image is done when the expert labeled it in this round or rejected it
(not a leaf); that is the count the label step waits for (lambdas/labeling.py).
"""

import json
from typing import Annotated
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from psycopg.rows import dict_row

from api.auth import User, require_groups
from api.deps import DataStore, Settings, get_data_store, get_db, get_settings
from common.admin import ActiveRound, ActiveRoundsResponse, RoundPhase, RoundStatus
from common.s3keys import selection_key

router = APIRouter()

ACTIVE = ("pending", "running")

_ROUNDS = """
    SELECT r.id, d.name AS dataset, r.status, r.strategy, r.k, bv.version AS base_version,
           r.started_by, r.commit_sha, r.ls_project_id, mv.mlflow_run_id,
           r.created_at, r.finished_at
      FROM rounds r
      JOIN datasets d ON d.id = r.dataset_id
      JOIN dataset_versions bv ON bv.id = r.base_version_id
      LEFT JOIN model_versions mv ON mv.round_id = r.id
"""

_DONE = """
    SELECT count(*) FROM images i
     WHERE i.id = ANY(%s)
       AND (i.status = 'rejected'
            OR EXISTS (SELECT 1 FROM labels l WHERE l.image_id = i.id AND l.round_id = %s))
"""


def _progress(
    db: psycopg.Connection, store: DataStore, round_id: UUID, status: str
) -> tuple[RoundPhase | None, int]:
    """The round's phase (None when finished) and how many selected images are done."""
    if status == "pending":
        return RoundPhase.STARTING, 0
    body = store.get(selection_key(round_id))
    if body is None:
        return (RoundPhase.TRAINING if status == "running" else None), 0
    ids = json.loads(body)["image_ids"]
    (done,) = db.execute(_DONE, (ids, round_id)).fetchone()
    if status != "running":
        return None, done
    return (RoundPhase.MERGING if done == len(ids) else RoundPhase.AWAITING_LABELS), done


def _label_studio_url(settings: Settings, project_id: int | None) -> str | None:
    return None if project_id is None else f"{settings.label_studio_url}/projects/{project_id}"


@router.get("/v1/rounds/active")
def active_rounds(
    _: Annotated[User, Depends(require_groups("labelers", "admins"))],
    db: Annotated[psycopg.Connection, Depends(get_db)],
    store: Annotated[DataStore, Depends(get_data_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ActiveRoundsResponse:
    """Rounds not finished yet, oldest first; a labeler finds their project here."""
    rows = db.cursor(row_factory=dict_row).execute(
        _ROUNDS + " WHERE r.status = ANY(%s) ORDER BY r.created_at, r.id", (list(ACTIVE),)
    )
    rounds = []
    for row in rows.fetchall():
        phase, labeled = _progress(db, store, row["id"], row["status"])
        rounds.append(
            ActiveRound(
                round_id=row["id"],
                dataset=row["dataset"],
                phase=phase,
                k=row["k"],
                labeled=labeled,
                label_studio_url=_label_studio_url(settings, row["ls_project_id"]),
            )
        )
    return ActiveRoundsResponse(rounds=rounds)


@router.get("/v1/rounds/{round_id}")
def round_status(
    round_id: UUID,
    _: Annotated[User, Depends(require_groups("admins"))],
    db: Annotated[psycopg.Connection, Depends(get_db)],
    store: Annotated[DataStore, Depends(get_data_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RoundStatus:
    row = db.cursor(row_factory=dict_row).execute(_ROUNDS + " WHERE r.id = %s", (round_id,))
    row = row.fetchone()
    if row is None:
        raise HTTPException(404, f"No round {round_id}.")
    phase, labeled = _progress(db, store, round_id, row["status"])
    run_id = row.pop("mlflow_run_id")
    del row["id"], row["ls_project_id"]
    return RoundStatus(
        round_id=round_id,
        phase=phase,
        labeled=labeled,
        mlflow_run_url=None if run_id is None else f"{settings.mlflow_url}/#/runs/{run_id}",
        **row,
    )
