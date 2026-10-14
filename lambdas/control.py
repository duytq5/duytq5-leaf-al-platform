"""Start and cancel rounds from GitHub (design doc F3, "Control through GitHub").

  start_round   invoked by the start-round workflow when a round config merges
                to main: validates it, inserts the round (pending) and returns
                the al-round execution input. The workflow then starts the
                execution named by the round id (this Lambda is in the VPC,
                which has no route to the Step Functions API).
  cancel_round  invoked by the cancel-round workflow after it stopped the
                execution: the round is cancelled and its queued images go back
                to the pool.

The round id is uuid5(commit SHA + config path), so a re-run workflow gets the
same round and starts nothing new.
"""

import os
from typing import Any
from uuid import UUID, uuid5

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, ValidationError

from common.round import RoundConfig, RoundExecution, TrainSection
from common.types import CommitSha, GitHubLogin, WorkerImage
from lambdas.logs import traced
from lambdas.postgres import open_db
from lambdas.round import S3Store, Store, end_round
from recipes import get_recipe
from selection import from_config

# Namespace of round ids (uuid5), fixed forever.
ROUND_NAMESPACE = UUID("8b0e7c64-5d1f-5b2e-a1c3-2f6d4e9a7b10")


class BadRound(ValueError):
    """The round config or request cannot start a round."""


class RoundConflict(RuntimeError):
    """The dataset already has an unfinished round."""


class StartRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]  # the round config file, validated below
    config_path: str
    image: WorkerImage  # the image of the platform commit the ops repo pins
    started_by: GitHubLogin  # who merged the round config
    commit_sha: CommitSha  # the ops repo commit that merged the round config


def round_uuid(commit_sha: str, config_path: str) -> UUID:
    return uuid5(ROUND_NAMESPACE, f"{commit_sha}:{config_path}")


def _validate(ev: StartRound):
    try:
        cfg = RoundConfig.model_validate(ev.config)
        recipe = None if cfg.train is None else get_recipe(cfg.train.recipe, cfg.train.params)
        strategy = from_config(cfg.selection)
    except KeyError as e:
        raise BadRound(e.args[0]) from None
    except ValidationError as e:
        raise BadRound(str(e)) from None
    return cfg, recipe, strategy


def _newest_model(conn: psycopg.Connection, dataset_id: int) -> str | None:
    row = conn.execute(
        """
        SELECT mlflow_name, mlflow_version FROM model_versions
         WHERE dataset_id = %s ORDER BY created_at DESC, id DESC LIMIT 1
        """,
        (dataset_id,),
    ).fetchone()
    return None if row is None else f"models:/{row[0]}/{row[1]}"


def start_round(conn: psycopg.Connection, ev: StartRound) -> dict[str, Any]:
    cfg, recipe, strategy = _validate(ev)
    round_id = round_uuid(ev.commit_sha, ev.config_path)

    dataset = conn.execute("SELECT id FROM datasets WHERE name = %s", (cfg.dataset,)).fetchone()
    if dataset is None:
        raise BadRound(f"no dataset {cfg.dataset}")
    train, score_model = None, None
    if recipe is not None:
        train = TrainSection(
            recipe=recipe.name,
            params=recipe.params.model_dump(mode="json"),
            base_model=cfg.train.base_model,
        )
    elif not strategy.requires:
        # A random round without `train` needs no scores: it scores only with
        # a named model (its scores become Label Studio predictions).
        score_model = cfg.model
    else:
        # A selection-only round (change 24) scores with an existing model.
        score_model = cfg.model or _newest_model(conn, dataset[0])
        if score_model is None:
            raise BadRound(
                f"{cfg.dataset} has no model to score with; add `train` or run a training first"
            )
    execution = RoundExecution(round_id=round_id, image=ev.image, train=train, model=score_model)
    out = {"round_id": str(round_id), "execution": execution.model_dump(mode="json")}

    if cfg.base_version is None:
        version = conn.execute(
            "SELECT id FROM dataset_versions WHERE dataset_id = %s ORDER BY version DESC LIMIT 1",
            (dataset[0],),
        ).fetchone()
    else:
        version = conn.execute(
            "SELECT id FROM dataset_versions WHERE dataset_id = %s AND version = %s",
            (dataset[0], cfg.base_version),
        ).fetchone()
    if version is None:
        raise BadRound(f"{cfg.dataset} has no version {cfg.base_version}")

    try:
        with conn.transaction():
            conn.execute(
                "INSERT INTO rounds (id, dataset_id, base_version_id, strategy, strategy_cfg,"
                " k, seed, started_by, commit_sha, score_model)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (id) DO NOTHING",
                (
                    round_id,
                    dataset[0],
                    version[0],
                    cfg.selection.strategy,
                    Jsonb(strategy.params.model_dump(mode="json")),
                    cfg.selection.k,
                    cfg.selection.seed,
                    ev.started_by,
                    ev.commit_sha,
                    score_model,
                ),
            )
    except psycopg.errors.UniqueViolation:
        (active,) = conn.execute(
            "SELECT id FROM rounds WHERE dataset_id = %s AND status IN ('pending', 'running')",
            (dataset[0],),
        ).fetchone()
        raise RoundConflict(
            f"{cfg.dataset} already has an unfinished round {active}; cancel it or wait"
        ) from None
    return out


def cancel_round(conn: psycopg.Connection, store: Store, round_id: UUID) -> dict[str, str]:
    row = conn.execute("SELECT 1 FROM rounds WHERE id = %s", (round_id,)).fetchone()
    if row is None:
        raise BadRound(f"no round {round_id}")
    end_round(conn, store, round_id, "cancelled")
    (status,) = conn.execute("SELECT status FROM rounds WHERE id = %s", (round_id,)).fetchone()
    return {"round_id": str(round_id), "status": status}


# --- Lambda handlers -------------------------------------------------------------


class _Cancel(BaseModel):
    round_id: UUID


def _store() -> S3Store:
    return S3Store(os.environ["DATA_BUCKET"])


@traced
def start_round_handler(event: dict, _context) -> dict:
    ev = StartRound.model_validate(event)
    with open_db() as conn:
        return start_round(conn, ev)


@traced
def cancel_round_handler(event: dict, _context) -> dict:
    round_id = _Cancel.model_validate(event).round_id
    with open_db() as conn:
        return cancel_round(conn, _store(), round_id)
