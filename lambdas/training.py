"""Standalone trainings (design doc change 24): train a model on a dataset
version without a round, so rounds can select and label several times before
the next training.

  start_training  invoked by the ops repo's start-training workflow when a
                  training file merges to main: validates it, inserts the
                  training (pending) and returns the al-train execution input.
                  The workflow then starts the execution named by the training
                  id (this Lambda is in the VPC, which has no route to the Step
                  Functions API).
  mark_running    pending -> running
  snapshot        writes the version manifest, builds the train job spec
  (train: the same SQS job and worker code as a round's)
  record          model_versions row, training succeeded
  fail            training failed

The training id is uuid5(commit SHA + config path), so a re-run workflow gets
the same training and starts nothing new. Every step is safe to run twice.
"""

import os
from typing import Any
from uuid import UUID, uuid5

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, ValidationError

from common.jobs import JobType, TrainResult, job_uuid
from common.round import TrainSection
from common.training import TrainingConfig, TrainingExecution
from common.types import WorkerImage
from lambdas.control import StartRound
from lambdas.logs import traced
from lambdas.postgres import open_db
from lambdas.round import RoundError, S3Store, _classes, train_spec
from recipes import get_recipe

# Namespace of training ids (uuid5), fixed forever.
TRAINING_NAMESPACE = UUID("3c9a1f52-7e4b-5d06-9a8e-6b2f0c1d4e73")


class BadTraining(ValueError):
    """The training config or request cannot start a training."""


class TrainingError(RuntimeError):
    """The training cannot go on; the state machine's catch marks it failed."""


class StartTraining(StartRound):
    """Same fields as a round start: the config file, its path, the worker
    image, the merger and the ops repo commit."""


def training_uuid(commit_sha: str, config_path: str) -> UUID:
    return uuid5(TRAINING_NAMESPACE, f"{commit_sha}:{config_path}")


def start_training(conn: psycopg.Connection, ev: StartTraining) -> dict[str, Any]:
    try:
        cfg = TrainingConfig.model_validate(ev.config)
        recipe = get_recipe(cfg.train.recipe, cfg.train.params)
    except KeyError as e:
        raise BadTraining(e.args[0]) from None
    except ValidationError as e:
        raise BadTraining(str(e)) from None
    train = TrainSection(
        recipe=recipe.name,
        params=recipe.params.model_dump(mode="json"),
        base_model=cfg.train.base_model,
    )
    dataset = conn.execute("SELECT id FROM datasets WHERE name = %s", (cfg.dataset,)).fetchone()
    if dataset is None:
        raise BadTraining(f"no dataset {cfg.dataset}")
    version = conn.execute(
        "SELECT id FROM dataset_versions WHERE dataset_id = %s"
        " AND (%s::int IS NULL OR version = %s) ORDER BY version DESC LIMIT 1",
        (dataset[0], cfg.version, cfg.version),
    ).fetchone()
    if version is None:
        raise BadTraining(f"{cfg.dataset} has no version {cfg.version}")

    training_id = training_uuid(ev.commit_sha, ev.config_path)
    conn.execute(
        "INSERT INTO trainings (id, dataset_id, version_id, train_config, started_by, commit_sha)"
        " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
        (
            training_id,
            dataset[0],
            version[0],
            Jsonb(train.model_dump(mode="json")),
            ev.started_by,
            ev.commit_sha,
        ),
    )
    execution = TrainingExecution(training_id=training_id, image=ev.image)
    return {"training_id": str(training_id), "execution": execution.model_dump(mode="json")}


_TRAINING_SQL = """
SELECT t.status, t.dataset_id, d.name, t.version_id, v.version, t.train_config
FROM trainings t
JOIN datasets d ON d.id = t.dataset_id
JOIN dataset_versions v ON v.id = t.version_id
WHERE t.id = %s
"""


def _load(conn: psycopg.Connection, training_id: UUID, *, lock: bool = False) -> tuple:
    row = conn.execute(
        _TRAINING_SQL + (" FOR UPDATE OF t" if lock else ""), (training_id,)
    ).fetchone()
    if row is None:
        raise TrainingError(f"training {training_id} does not exist")
    return row


def mark_running(conn: psycopg.Connection, training_id: UUID) -> dict[str, Any]:
    with conn.transaction():
        status, _, dataset, _, version, _ = _load(conn, training_id, lock=True)
        if status == "pending":
            conn.execute("UPDATE trainings SET status = 'running' WHERE id = %s", (training_id,))
        elif status != "running":
            raise TrainingError(f"training {training_id} is {status}; it cannot run")
    return {"dataset": dataset, "version": version}


def snapshot(conn: psycopg.Connection, store: S3Store, training_id: UUID) -> dict[str, Any]:
    """The version's manifest and the train job (same as a round's train job)."""
    status, dataset_id, dataset, version_id, version, train = _load(conn, training_id)
    if status != "running":
        raise TrainingError(f"training {training_id} is {status}, not running")
    try:
        spec = train_spec(
            conn, store, TrainSection.model_validate(train), dataset_id, dataset, version_id,
            version,
        )  # fmt: skip
    except RoundError as e:
        raise TrainingError(str(e)) from None
    return {
        "job_id": str(job_uuid(training_id, JobType.TRAIN)),
        "spec": spec.model_dump(mode="json"),
    }


def record(
    conn: psycopg.Connection, training_id: UUID, result: TrainResult, image: str
) -> dict[str, str]:
    """Store the trained model (with the worker image it was trained in, for
    the release's export) and finish the training, in one transaction."""
    with conn.transaction():
        status, dataset_id, _, version_id, _, train = _load(conn, training_id, lock=True)
        if status == "running":
            conn.execute(
                """
                INSERT INTO model_versions (training_id, dataset_id, trained_on_version_id,
                    mlflow_name, mlflow_version, mlflow_run_id, arch, train_config, labels,
                    metrics, image)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    training_id,
                    dataset_id,
                    version_id,
                    result.mlflow_name,
                    result.mlflow_version,
                    result.mlflow_run_id,
                    result.arch,
                    Jsonb(train),
                    Jsonb(_classes(conn, dataset_id)),
                    Jsonb(result.metrics),
                    image,
                ),
            )
            conn.execute(
                "UPDATE trainings SET status = 'succeeded', finished_at = now() WHERE id = %s",
                (training_id,),
            )
        elif status != "succeeded":
            raise TrainingError(f"training {training_id} is {status}, not running")
    name, version = conn.execute(
        "SELECT mlflow_name, mlflow_version FROM model_versions WHERE training_id = %s",
        (training_id,),
    ).fetchone()
    return {"model": f"models:/{name}/{version}"}


def fail(conn: psycopg.Connection, training_id: UUID) -> None:
    """Only an unfinished training changes, so a second call changes nothing."""
    conn.execute(
        "UPDATE trainings SET status = 'failed', finished_at = now()"
        " WHERE id = %s AND status IN ('pending', 'running')",
        (training_id,),
    )


# --- Lambda handlers -------------------------------------------------------------


class _Event(BaseModel):
    model_config = ConfigDict(extra="ignore")
    training_id: UUID


class _RecordEvent(_Event):
    train_result: TrainResult
    image: WorkerImage


def _store() -> S3Store:
    return S3Store(os.environ["DATA_BUCKET"])


@traced
def start_training_handler(event: dict, _context) -> dict:
    ev = StartTraining.model_validate(event)
    with open_db() as conn:
        return start_training(conn, ev)


@traced
def mark_running_handler(event: dict, _context) -> dict:
    with open_db() as conn:
        return mark_running(conn, _Event.model_validate(event).training_id)


@traced
def snapshot_handler(event: dict, _context) -> dict:
    with open_db() as conn:
        return snapshot(conn, _store(), _Event.model_validate(event).training_id)


@traced
def record_handler(event: dict, _context) -> dict:
    ev = _RecordEvent.model_validate(event)
    with open_db() as conn:
        return record(conn, ev.training_id, ev.train_result, ev.image)


@traced
def fail_handler(event: dict, _context) -> None:
    with open_db() as conn:
        fail(conn, _Event.model_validate(event).training_id)
