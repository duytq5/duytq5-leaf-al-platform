"""Job dispatch inside the worker image: one handler per job type, each
returning the output sent back with SendTaskSuccess. The work is in
worker.compute; the environment (S3, cache, models bucket) comes from the
container's environment, set by the agent."""

from collections.abc import Callable
from functools import cache

from common.jobs import ExportJob, JobType, ScoreJob, TrainJob

Job = TrainJob | ScoreJob | ExportJob


@cache
def _env():  # pragma: no cover - wiring only
    from worker.compute import WorkerEnv

    return WorkerEnv.from_environ()


def run_train(job: TrainJob) -> dict:
    from worker.compute import run_train

    return run_train(job, _env())


def run_score(job: ScoreJob) -> dict:
    from worker.compute import run_score

    return run_score(job, _env())


def run_export(job: ExportJob) -> dict:
    from worker.compute import run_export

    return run_export(job, _env())


HANDLERS: dict[JobType, Callable[..., dict]] = {
    JobType.TRAIN: run_train,
    JobType.SCORE: run_score,
    JobType.EXPORT: run_export,
}


def dispatch(job: Job) -> dict:
    """Run a job and return the output sent back with SendTaskSuccess."""
    return HANDLERS[job.type](job)
