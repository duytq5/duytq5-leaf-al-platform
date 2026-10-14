"""The worker's log lines go to CloudWatch (/leaf-al/worker) as JSON, each
tagged with the job it belongs to, so `al logs <round_id>` finds them."""

import json
import logging
import uuid

from botocore.exceptions import ClientError

from worker.logs import CloudWatchHandler, JsonFormatter, job_context, job_fields

ROUND_ID = uuid.uuid4()
JOB_ID = uuid.uuid4()


class FakeLogs:
    def __init__(self, fail_put: bool = False, stream_exists: bool = False) -> None:
        self.streams: list[tuple[str, str]] = []
        self.batches: list[list[dict]] = []
        self.fail_put, self.stream_exists = fail_put, stream_exists

    def create_log_stream(self, logGroupName, logStreamName):
        self.streams.append((logGroupName, logStreamName))
        if self.stream_exists:
            raise ClientError(
                {"Error": {"Code": "ResourceAlreadyExistsException", "Message": "x"}}, "op"
            )

    def put_log_events(self, logGroupName, logStreamName, logEvents):
        assert (logGroupName, logStreamName) == ("/leaf-al/worker", "gpu-1")
        if self.fail_put:
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "op")
        self.batches.append(logEvents)


def _record(msg: str, level=logging.INFO, exc_info=None) -> logging.LogRecord:
    return logging.LogRecord("worker.agent", level, __file__, 1, msg, (), exc_info)


def _handler(client, **kw) -> CloudWatchHandler:
    handler = CloudWatchHandler(client, "/leaf-al/worker", "gpu-1", **kw)
    handler.setFormatter(JsonFormatter())
    return handler


class _Job:
    job_id = JOB_ID
    round_id = ROUND_ID
    type = "train"


def test_job_fields():
    assert job_fields(_Job()) == {
        "job_id": str(JOB_ID),
        "round_id": str(ROUND_ID),
        "job_type": "train",
    }

    class Export(_Job):
        round_id = None
        type = "export"

    assert "round_id" not in job_fields(Export())

    class Training(_Job):
        round_id = None
        training_id = ROUND_ID

    assert job_fields(Training())["training_id"] == str(ROUND_ID)


def test_json_lines_carry_the_running_job():
    fmt = JsonFormatter()
    assert "round_id" not in json.loads(fmt.format(_record("idle")))
    token = job_context.set(job_fields(_Job()))
    try:
        line = json.loads(fmt.format(_record("epoch 3 val_f1=0.81")))
    finally:
        job_context.reset(token)
    assert line["message"] == "epoch 3 val_f1=0.81"
    assert line["level"] == "INFO"
    assert line["logger"] == "worker.agent"
    assert line["round_id"] == str(ROUND_ID) and line["job_id"] == str(JOB_ID)


def test_json_lines_keep_the_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        line = json.loads(JsonFormatter().format(_record("failed", logging.ERROR, sys.exc_info())))
    assert line["level"] == "ERROR"
    assert "ValueError: boom" in line["exception"]


def test_flush_sends_buffered_lines_in_order():
    client = FakeLogs()
    handler = _handler(client)
    handler.open()
    handler.emit(_record("a"))
    handler.emit(_record("b"))
    handler.flush()
    handler.flush()  # nothing new: no empty call
    assert client.streams == [("/leaf-al/worker", "gpu-1")]
    (batch,) = client.batches
    assert [json.loads(e["message"])["message"] for e in batch] == ["a", "b"]
    assert all(isinstance(e["timestamp"], int) for e in batch)


def test_an_existing_stream_is_fine():
    handler = _handler(FakeLogs(stream_exists=True))
    handler.open()


def test_big_buffers_go_in_several_calls():
    client = FakeLogs()
    handler = _handler(client, max_events=3)
    handler.open()
    for i in range(7):
        handler.emit(_record(str(i)))
    handler.flush()
    assert [len(b) for b in client.batches] == [3, 3, 1]


def test_huge_lines_are_cut():
    client = FakeLogs()
    handler = _handler(client)
    handler.open()
    handler.emit(_record("x" * 400_000))
    handler.flush()
    ((event,),) = client.batches
    assert len(event["message"].encode()) <= CloudWatchHandler.MAX_EVENT_BYTES


def test_cloudwatch_errors_never_stop_the_worker(capsys):
    client = FakeLogs(fail_put=True)
    handler = _handler(client)
    handler.open()
    handler.emit(_record("a"))
    handler.flush()  # no exception
    handler.emit(_record("b"))
    handler.flush()
    err = capsys.readouterr().err
    assert err.count("CloudWatch") == 1  # said once, not on every flush


def test_close_flushes_and_stops_the_timer():
    client = FakeLogs()
    handler = _handler(client, flush_every=60)
    handler.open()
    handler.emit(_record("last"))
    handler.close()
    assert len(client.batches) == 1
    assert not handler._thread.is_alive()


def test_the_agent_tags_lines_with_the_job_while_it_runs():
    from tests.test_worker_agent import FakeSfn, FakeSqs, _agent, _job

    seen = []
    agent = _agent(FakeSqs(_job()), FakeSfn(), lambda job: seen.append(job_context.get()) or {})
    agent.poll_once()
    (fields,) = seen
    assert fields["job_type"] == "export" and "job_id" in fields
    assert not job_context.get()  # cleared after the job


def test_container_output_is_logged_line_by_line(tmp_path, caplog):
    from common.jobs import parse_job
    from tests.test_worker_agent import FakeRun, _job, _runner

    caplog.set_level(logging.INFO, logger="worker.job")
    run = FakeRun(result={"onnx_sha256": "ab" * 32}, logs="epoch 1\nepoch 2\n")
    _runner(tmp_path, run).run(parse_job(json.dumps(_job())))
    assert [r.getMessage() for r in caplog.records if r.name == "worker.job"] == [
        "epoch 1",
        "epoch 2",
    ]
