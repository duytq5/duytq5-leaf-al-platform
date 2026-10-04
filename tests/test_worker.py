from common.jobs import JobType
from worker.jobs import HANDLERS


def test_every_job_type_has_a_handler():
    assert set(HANDLERS) == set(JobType)
