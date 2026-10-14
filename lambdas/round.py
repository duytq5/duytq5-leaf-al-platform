"""Lambda steps of the al-round state machine (design doc F4).

  mark_running  pending -> running; returns the round settings
  snapshot      writes the version manifest and the pool list, builds the
                train and score job specs (no train job for a selection-only
                round, design doc change 24: it scores with the round's model)
  (train, score: SQS jobs the worker runs; Step Functions waits for them)
  select        runs the round's strategy on the scores, marks k images queued
  (await labels: lambdas/labeling.py puts the selected images into Label
                Studio; its webhook resumes the round once all k are labeled, F2)
  merge         new dataset version, model_versions row (if the round
                trained), round succeeded
  fail          round failed, its queued images back to unlabeled

Step Functions retries a step after a timeout or a crash, so every step is
safe to run twice: writes are guarded by the round's current state, files
are rewritten whole, and the selection is read back from S3 when it exists.
"""

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import boto3
import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict

from common.jobs import JobType, TrainResult, TrainSpec, job_uuid
from common.round import RoundExecution, TrainSection
from common.s3keys import manifest_key, pool_manifest_key, raw_key, scores_key, selection_key
from lambdas.logs import traced
from lambdas.postgres import open_db
from selection.base import LABELED_COUNTS_SQL, LabeledStats, PoolData, get_strategy


class RoundError(RuntimeError):
    """The round cannot go on; the state machine's catch marks it failed."""


class Store(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, body: bytes) -> None: ...


class S3Store:
    def __init__(self, bucket: str, client=None) -> None:
        self.bucket = bucket
        self.client = client or boto3.client("s3")

    def get(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def put(self, key: str, body: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"


@dataclass(frozen=True)
class Round:
    id: UUID
    dataset_id: int
    dataset: str
    base_version_id: int
    base_version: int
    strategy: str
    strategy_cfg: dict[str, Any]
    k: int
    seed: int
    status: str


_ROUND_SQL = """
SELECT r.id, r.dataset_id, d.name, r.base_version_id, v.version,
       r.strategy, r.strategy_cfg, r.k, r.seed, r.status
FROM rounds r
JOIN datasets d ON d.id = r.dataset_id
JOIN dataset_versions v ON v.id = r.base_version_id
WHERE r.id = %s
"""


def load_round(conn: psycopg.Connection, round_id: UUID, *, lock: bool = False) -> Round:
    row = conn.execute(_ROUND_SQL + (" FOR UPDATE OF r" if lock else ""), (round_id,)).fetchone()
    if row is None:
        raise RoundError(f"round {round_id} does not exist")
    return Round(*row)


def _running(conn: psycopg.Connection, round_id: UUID) -> Round:
    rnd = load_round(conn, round_id)
    if rnd.status != "running":
        raise RoundError(f"round {round_id} is {rnd.status}, not running")
    return rnd


def _classes(conn: psycopg.Connection, dataset_id: int) -> list[dict[str, str]]:
    rows = conn.execute(
        "SELECT code, display_name FROM classes WHERE dataset_id = %s ORDER BY position",
        (dataset_id,),
    ).fetchall()
    return [{"code": code, "display_name": name} for code, name in rows]


def _parquet(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _selection(store: Store, round_id: UUID) -> list[int] | None:
    body = store.get(selection_key(round_id))
    return None if body is None else json.loads(body)["image_ids"]


# --- steps ---------------------------------------------------------------------


def mark_running(conn: psycopg.Connection, round_id: UUID) -> dict[str, Any]:
    with conn.transaction():
        rnd = load_round(conn, round_id, lock=True)
        if rnd.status == "pending":
            conn.execute("UPDATE rounds SET status = 'running' WHERE id = %s", (round_id,))
        elif rnd.status != "running":
            raise RoundError(f"round {round_id} is {rnd.status}; it cannot run")
    return {"dataset": rnd.dataset, "base_version": rnd.base_version, "k": rnd.k}


def snapshot(conn: psycopg.Connection, store: S3Store, run: RoundExecution) -> dict[str, Any]:
    """Write what the worker reads: the base version's labeled images (all
    splits; the worker trains on train, picks the checkpoint on val and reports
    test) and the unlabeled pool. Returns the specs of the train and score jobs.
    A selection-only round gets no manifest and no train job; its score job
    uses the round's model."""
    rnd = _running(conn, run.round_id)
    out: dict[str, Any] = {}
    if run.train is not None:
        out["train"] = _train_job(conn, store, run, rnd)

    pool = conn.execute(
        """
        SELECT id, sha256 FROM images
        WHERE dataset_id = %s AND split = 'train' AND status = 'unlabeled'
        ORDER BY id
        """,
        (rnd.dataset_id,),
    ).fetchall()
    if not pool:
        raise RoundError(f"dataset {rnd.dataset} has no unlabeled images to select from")
    pool_key = pool_manifest_key(run.round_id)
    store.put(
        pool_key,
        _parquet(
            pa.table(
                {
                    "image_id": [r[0] for r in pool],
                    "sha256": [r[1] for r in pool],
                    "key": [raw_key(rnd.dataset, r[1]) for r in pool],
                }
            )
        ),
    )

    strategy = get_strategy(rnd.strategy, seed=rnd.seed, params=rnd.strategy_cfg)
    out["score"] = {
        "job_id": str(job_uuid(run.round_id, JobType.SCORE)),
        "pool_manifest_uri": store.uri(pool_key),
        "outputs": sorted(strategy.requires | {"probs"}),
        "output_uri": store.uri(scores_key(run.round_id)),
    }
    # A round that trains scores with the model the train job returns; the
    # state machine passes it on.
    if run.model is not None:
        out["score"]["model_uri"] = run.model
    out["pool_size"] = len(pool)
    return out


def _train_job(
    conn: psycopg.Connection, store: S3Store, run: RoundExecution, rnd: Round
) -> dict[str, Any]:
    """The base version's manifest and the train job's spec."""
    return {
        "job_id": str(job_uuid(run.round_id, JobType.TRAIN)),
        "spec": train_spec(
            conn, store, run.train, rnd.dataset_id, rnd.dataset, rnd.base_version_id,
            rnd.base_version,
        ).model_dump(mode="json"),
    }  # fmt: skip


def train_spec(
    conn: psycopg.Connection,
    store: S3Store,
    train: TrainSection,
    dataset_id: int,
    dataset: str,
    version_id: int,
    version: int,
) -> TrainSpec:
    """Write the version's manifest (its labeled images, all splits: the worker
    trains on train, picks the checkpoint on val and reports test) and return
    the train job's spec. Also used by standalone trainings (lambdas/training.py)."""
    labeled = conn.execute(
        """
        SELECT i.id, i.sha256, i.split, c.code
        FROM dataset_version_labels v
        JOIN labels l ON l.id = v.label_id
        JOIN classes c ON c.id = l.class_id
        JOIN images i ON i.id = v.image_id
        WHERE v.version_id = %s
        ORDER BY i.id
        """,
        (version_id,),
    ).fetchall()
    if not any(split == "train" for _, _, split, _ in labeled):
        raise RoundError(f"version v{version} has no labeled train images")
    manifest = manifest_key(dataset, version)
    store.put(
        manifest,
        _parquet(
            pa.table(
                {
                    "image_id": [r[0] for r in labeled],
                    "sha256": [r[1] for r in labeled],
                    "key": [raw_key(dataset, r[1]) for r in labeled],
                    "split": [r[2] for r in labeled],
                    "class_code": [r[3] for r in labeled],
                }
            )
        ),
    )
    conn.execute(
        "UPDATE dataset_versions SET manifest_uri = %s WHERE id = %s AND manifest_uri IS NULL",
        (store.uri(manifest), version_id),
    )
    return TrainSpec(
        **train.model_dump(),
        dataset=dataset,
        manifest_uri=store.uri(manifest),
        classes=_classes(conn, dataset_id),
    )


def read_scores(body: bytes) -> PoolData:
    table = pq.read_table(io.BytesIO(body))
    ids = [str(i) for i in table.column("image_id").to_pylist()]
    probs = np.array(table.column("probs").to_pylist(), dtype=np.float64)
    return PoolData(ids=ids, probs=probs.reshape(len(ids), -1))


def select(conn: psycopg.Connection, store: Store, round_id: UUID) -> dict[str, Any]:
    """Pick k images with the round's strategy and mark them queued. The first
    run writes selections/{round_id}.json; a retry reuses it, so a strategy with
    randomness cannot pick a second, different set."""
    rnd = _running(conn, round_id)
    ids = _selection(store, round_id)
    if ids is None:
        body = store.get(scores_key(round_id))
        if body is None:
            raise RoundError(f"no scores at {scores_key(round_id)}")
        pool = read_scores(body)
        counts = dict(
            conn.execute(LABELED_COUNTS_SQL, {"version_id": rnd.base_version_id}).fetchall()
        )
        codes = [c["code"] for c in _classes(conn, rnd.dataset_id)]
        strategy = get_strategy(rnd.strategy, seed=rnd.seed, params=rnd.strategy_cfg)
        chosen = strategy.select(pool, LabeledStats.from_counts(counts, codes), rnd.k)
        # Only images still in the pool: one that was labeled or rejected
        # since the score job ran is skipped.
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM images WHERE id = ANY(%s) AND dataset_id = %s"
                " AND split = 'train' AND status = 'unlabeled' ORDER BY id",
                ([int(i) for i in chosen], rnd.dataset_id),
            ).fetchall()
        ]
        if not ids:
            raise RoundError("the strategy selected no image from the pool")
        store.put(
            selection_key(round_id),
            json.dumps({"round_id": str(round_id), "image_ids": ids}).encode(),
        )
    conn.execute(
        "UPDATE images SET status = 'queued' WHERE id = ANY(%s) AND status = 'unlabeled'",
        (ids,),
    )
    return {"selected": len(ids)}


def merge(
    conn: psycopg.Connection,
    store: Store,
    run: RoundExecution,
    result: TrainResult | None,
) -> dict[str, Any]:
    """Make the round's dataset version (its base plus this round's labels),
    record the model it trained (a selection-only round has none), and finish
    the round, in one transaction."""
    if run.train is not None and result is None:
        raise RoundError(f"round {run.round_id} trains but has no train result")
    ids = _selection(store, run.round_id) or []
    with conn.transaction():
        rnd = load_round(conn, run.round_id, lock=True)
        if rnd.status == "succeeded":
            return _merged(conn, run.round_id)
        if rnd.status != "running":
            raise RoundError(f"round {run.round_id} is {rnd.status}, not running")
        (version,) = conn.execute(
            """
            INSERT INTO dataset_versions (dataset_id, version, parent_version_id,
                                          created_by_round_id)
            SELECT %s, max(version) + 1, %s, %s FROM dataset_versions WHERE dataset_id = %s
            RETURNING id
            """,
            (rnd.dataset_id, rnd.base_version_id, run.round_id, rnd.dataset_id),
        ).fetchone()
        # The round's own labels win over the base version's for the same image.
        conn.execute(
            """
            INSERT INTO dataset_version_labels (version_id, image_id, label_id)
            SELECT %(v)s, l.image_id, max(l.id) FROM labels l
            WHERE l.round_id = %(r)s GROUP BY l.image_id
            UNION ALL
            SELECT %(v)s, b.image_id, b.label_id FROM dataset_version_labels b
            WHERE b.version_id = %(base)s
              AND NOT EXISTS (SELECT 1 FROM labels l
                              WHERE l.round_id = %(r)s AND l.image_id = b.image_id)
            """,
            {"v": version, "r": run.round_id, "base": rnd.base_version_id},
        )
        conn.execute(
            """
            UPDATE images i SET status = CASE
                WHEN EXISTS (SELECT 1 FROM labels l
                             WHERE l.round_id = %s AND l.image_id = i.id) THEN 'labeled'
                ELSE 'unlabeled' END
            WHERE i.id = ANY(%s) AND i.status = 'queued'
            """,
            (run.round_id, ids),
        )
        if run.train is not None:
            conn.execute(
                """
                INSERT INTO model_versions (round_id, dataset_id, trained_on_version_id,
                    mlflow_name, mlflow_version, mlflow_run_id, arch, train_config, labels,
                    metrics, image)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run.round_id,
                    rnd.dataset_id,
                    rnd.base_version_id,
                    result.mlflow_name,
                    result.mlflow_version,
                    result.mlflow_run_id,
                    result.arch,
                    Jsonb(run.train.model_dump(mode="json")),
                    Jsonb(_classes(conn, rnd.dataset_id)),
                    Jsonb(result.metrics),
                    run.image,
                ),
            )
        conn.execute(
            "UPDATE rounds SET status = 'succeeded', metrics = %s, finished_at = now(),"
            " label_task_token = NULL WHERE id = %s",
            (None if result is None else Jsonb(result.metrics), run.round_id),
        )
    return _merged(conn, run.round_id)


def _merged(conn: psycopg.Connection, round_id: UUID) -> dict[str, Any]:
    version, labels = conn.execute(
        """
        SELECT v.version, count(vl.id) FROM dataset_versions v
        LEFT JOIN dataset_version_labels vl ON vl.version_id = v.id
        WHERE v.created_by_round_id = %s GROUP BY v.version
        """,
        (round_id,),
    ).fetchone()
    return {"version": version, "labels": labels}


def fail(conn: psycopg.Connection, store: Store, round_id: UUID) -> None:
    """Round failed: its queued images go back to the pool."""
    end_round(conn, store, round_id, "failed")


def end_round(conn: psycopg.Connection, store: Store, round_id: UUID, status: str) -> None:
    """Round failed or cancelled: its queued images go back to the pool. Only
    an unfinished round and images still queued are touched, so a second call
    changes nothing."""
    ids = _selection(store, round_id) or []
    with conn.transaction():
        conn.execute(
            "UPDATE rounds SET status = %s, finished_at = now(), label_task_token = NULL"
            " WHERE id = %s AND status IN ('pending', 'running')",
            (status, round_id),
        )
        conn.execute(
            "UPDATE images SET status = 'unlabeled' WHERE id = ANY(%s) AND status = 'queued'",
            (ids,),
        )


# --- Lambda handlers -------------------------------------------------------------


class _Event(BaseModel):
    model_config = ConfigDict(extra="ignore")
    round_id: UUID


class _MergeEvent(BaseModel):
    """The Merge state's whole input: the execution input plus each step's
    result; train_result only when the round trained."""

    model_config = ConfigDict(extra="ignore")
    round_id: UUID
    image: str
    train: dict[str, Any] | None = None
    model: str | None = None
    train_result: TrainResult | None = None


def _store() -> S3Store:
    return S3Store(os.environ["DATA_BUCKET"])


@traced
def mark_running_handler(event: dict, _context) -> dict:
    with open_db() as conn:
        return mark_running(conn, _Event.model_validate(event).round_id)


@traced
def snapshot_handler(event: dict, _context) -> dict:
    run = RoundExecution.model_validate(event)
    with open_db() as conn:
        return snapshot(conn, _store(), run)


@traced
def select_handler(event: dict, _context) -> dict:
    with open_db() as conn:
        return select(conn, _store(), _Event.model_validate(event).round_id)


@traced
def merge_handler(event: dict, _context) -> dict:
    ev = _MergeEvent.model_validate(event)
    run = RoundExecution(round_id=ev.round_id, image=ev.image, train=ev.train, model=ev.model)
    with open_db() as conn:
        return merge(conn, _store(), run, ev.train_result)


@traced
def fail_handler(event: dict, _context) -> None:
    with open_db() as conn:
        fail(conn, _store(), _Event.model_validate(event).round_id)
