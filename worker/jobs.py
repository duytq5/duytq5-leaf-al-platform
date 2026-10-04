"""Job dispatch. Handlers are implemented in Phase 1 (train, score) and Phase 3 (export)."""

from collections.abc import Callable

from common.jobs import ExportJob, JobType, ScoreJob, TrainJob

Job = TrainJob | ScoreJob | ExportJob


def run_train(job: TrainJob) -> dict:
    raise NotImplementedError("train job: Phase 1")


def run_score(job: ScoreJob) -> dict:
    raise NotImplementedError("score job: Phase 1")


def run_export(job: ExportJob) -> dict:
    raise NotImplementedError("export job: Phase 3")


HANDLERS: dict[JobType, Callable[..., dict]] = {
    JobType.TRAIN: run_train,
    JobType.SCORE: run_score,
    JobType.EXPORT: run_export,
}


def dispatch(job: Job) -> dict:
    """Run a job and return the output sent back with SendTaskSuccess."""
    return HANDLERS[job.type](job)
