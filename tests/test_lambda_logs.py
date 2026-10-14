import logging

import pytest

import lambdas.round as round_steps
from lambdas import labeling, release
from lambdas.logs import traced

ROUND_ID = "0b9e6f9c-6a51-4c7e-9d0f-1f2a3b4c5d6e"


def test_logs_the_ids_of_the_event(caplog):
    @traced
    def handler(event, _context):
        return "ok"

    caplog.set_level(logging.INFO)
    event = {"round_id": ROUND_ID, "task_token": "secret-token", "image": "x"}
    assert handler(event, None) == "ok"
    (record,) = caplog.records
    assert record.round_id == ROUND_ID
    assert ROUND_ID in record.getMessage()
    assert "handler" in record.getMessage()
    assert "secret-token" not in record.getMessage()
    assert not hasattr(record, "task_token")


def test_release_ids(caplog):
    @traced
    def handler(event, _context):
        return None

    caplog.set_level(logging.INFO)
    handler({"dataset": "rocole", "model": "m", "version": 3, "commit_sha": "abc"}, None)
    (record,) = caplog.records
    assert (record.dataset, record.model, record.version) == ("rocole", "m", 3)
    assert "rocole" in record.getMessage()


def test_logs_a_failure_with_the_ids_and_raises(caplog):
    @traced
    def handler(event, _context):
        raise ValueError("boom")

    caplog.set_level(logging.INFO)
    with pytest.raises(ValueError, match="boom"):
        handler({"round_id": ROUND_ID}, None)
    failed = caplog.records[-1]
    assert failed.levelno == logging.ERROR
    assert failed.exc_info is not None
    assert failed.round_id == ROUND_ID
    assert ROUND_ID in failed.getMessage()


def test_events_that_are_not_dicts_pass_through(caplog):
    @traced
    def handler(event, _context):
        return event

    caplog.set_level(logging.INFO)
    assert handler([1, 2], None) == [1, 2]


@pytest.mark.parametrize(
    "fn",
    [
        round_steps.mark_running_handler,
        round_steps.snapshot_handler,
        round_steps.select_handler,
        round_steps.merge_handler,
        round_steps.fail_handler,
        labeling.start_handler,
        labeling.record_handler,
        release.prepare_handler,
        release.record_handler,
    ],
)
def test_round_and_release_handlers_are_traced(fn):
    assert getattr(fn, "traced", False) is True
