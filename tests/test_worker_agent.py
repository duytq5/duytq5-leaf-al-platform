"""The worker agent on the GPU machine (design doc "Worker"): long-polls the
job queue, runs each job in the worker image the job names, and reports the
result to Step Functions with the job's task token.

SQS and Step Functions are in-memory fakes; the job runner is injected.
"""

import json
import threading
import time
import uuid

import pytest
from botocore.exceptions import ClientError

from worker.agent import Agent, DockerRunner, JobFailed

COMMIT = "a" * 40
IMAGE = f"ghcr.io/duytq5/duytq5-leaf-al-platform-worker:{COMMIT}"


def _job(task_token="tok-1", **spec) -> dict:
    return {
        "job_id": str(uuid.uuid4()),
        "type": "export",
        "image": IMAGE,
        "task_token": task_token,
        "spec": {
            "model_uri": "models:/rocole-fastvit/2",
            "output_prefix": "edge/rocole-fastvit/2/",
            "labels": [{"code": "a", "display_name": "A"}, {"code": "b", "display_name": "B"}],
            **spec,
        },
    }


def _error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "op")


class FakeSqs:
    def __init__(self, *bodies) -> None:
        self.queue = [
            {
                "MessageId": f"m{i}",
                "ReceiptHandle": f"r{i}",
                "Body": b if isinstance(b, str) else json.dumps(b),
            }
            for i, b in enumerate(bodies)
        ]
        self.deleted: list[str] = []
        self.extended: list[tuple[str, int]] = []

    def receive_message(self, **kw):
        assert kw["WaitTimeSeconds"] == 20  # long polling
        out, self.queue = (
            self.queue[: kw["MaxNumberOfMessages"]],
            self.queue[kw["MaxNumberOfMessages"] :],
        )
        return {"Messages": out} if out else {}

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted.append(ReceiptHandle)

    def change_message_visibility(self, QueueUrl, ReceiptHandle, VisibilityTimeout):
        self.extended.append((ReceiptHandle, VisibilityTimeout))


class FakeSfn:
    def __init__(self, error: str | None = None) -> None:
        self.calls: list[tuple] = []
        self.error = error

    def send_task_success(self, taskToken, output):
        self.calls.append(("success", taskToken, json.loads(output)))
        if self.error:
            raise _error(self.error)

    def send_task_failure(self, taskToken, error, cause):
        self.calls.append(("failure", taskToken, error, cause))
        if self.error:
            raise _error(self.error)


def _agent(sqs, sfn, runner, **kw) -> Agent:
    return Agent(sqs=sqs, sfn=sfn, queue_url="q", runner=runner, **kw)


def test_a_finished_job_reports_success_and_leaves_the_queue():
    job = _job()
    sqs, sfn = FakeSqs(job), FakeSfn()
    seen = []

    def runner(parsed):
        seen.append(parsed)
        return {"onnx_s3_key": "edge/x/2/model.onnx"}

    assert _agent(sqs, sfn, runner).poll_once() == 1
    assert str(seen[0].job_id) == job["job_id"]
    assert sfn.calls == [("success", "tok-1", {"onnx_s3_key": "edge/x/2/model.onnx"})]
    assert sqs.deleted == ["r0"]


def test_a_failed_job_reports_failure_and_leaves_the_queue():
    sqs, sfn = FakeSqs(_job()), FakeSfn()

    def runner(parsed):
        raise JobFailed("ONNX and PyTorch disagree on 3 of 120 validation images")

    _agent(sqs, sfn, runner).poll_once()
    ((kind, token, error, cause),) = sfn.calls
    assert (kind, token, error) == ("failure", "tok-1", "JobFailed")
    assert "disagree" in cause
    assert sqs.deleted == ["r0"]


def test_a_crash_in_the_job_is_a_failure_with_a_short_cause():
    sqs, sfn = FakeSqs(_job()), FakeSfn()

    def runner(parsed):
        raise RuntimeError("x" * 100_000)

    _agent(sqs, sfn, runner).poll_once()
    (_, _, error, cause) = sfn.calls[0]
    assert error == "RuntimeError"
    assert len(cause) <= 32_768  # Step Functions' limit


def test_a_message_that_is_not_a_job_fails_its_task_if_it_has_a_token():
    bad = _job()
    bad["type"] = "evaluate"
    sqs, sfn = FakeSqs(bad, "not json"), FakeSfn()
    agent = _agent(sqs, sfn, lambda j: pytest.fail("must not run"))
    # One job at a time: there is one GPU.
    assert (agent.poll_once(), agent.poll_once()) == (1, 1)
    assert [c[:3] for c in sfn.calls] == [("failure", "tok-1", "InvalidJob")]
    assert sqs.deleted == ["r0", "r1"]  # never retried: it cannot get better


@pytest.mark.parametrize("code", ["TaskTimedOut", "TaskDoesNotExist", "InvalidToken"])
def test_a_job_the_state_machine_gave_up_on_is_dropped(code):
    sqs, sfn = FakeSqs(_job()), FakeSfn(error=code)
    _agent(sqs, sfn, lambda j: {"ok": 1}).poll_once()
    assert sqs.deleted == ["r0"]


def test_a_report_that_does_not_reach_step_functions_is_retried_later():
    sqs, sfn = FakeSqs(_job()), FakeSfn(error="ThrottlingException")
    _agent(sqs, sfn, lambda j: {"ok": 1}).poll_once()
    assert sqs.deleted == []  # SQS hands the job out again after its visibility timeout


def test_a_long_job_keeps_its_message_hidden():
    sqs, sfn = FakeSqs(_job()), FakeSfn()
    done = threading.Event()

    def runner(parsed):
        done.wait(0.5)
        return {}

    agent = _agent(sqs, sfn, runner, extend_every=0.05, visibility_timeout=1800)
    agent.poll_once()
    assert len(sqs.extended) >= 3
    assert set(sqs.extended) == {("r0", 1800)}
    n = len(sqs.extended)
    time.sleep(0.15)
    assert len(sqs.extended) == n  # stops once the job is done


def test_an_empty_poll():
    assert _agent(FakeSqs(), FakeSfn(), lambda j: {}).poll_once() == 0


# Running a job in its worker image


class FakeRun:
    def __init__(self, returncode=0, result: dict | None = None, logs="") -> None:
        self.returncode, self.result, self.logs = returncode, result, logs
        self.commands: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.commands.append(cmd)
        self.kw = kw
        out = next(c.split(":")[0] for c in cmd if c.endswith(":/out"))
        if self.result is not None:
            with open(f"{out}/result.json", "w") as f:
                json.dump(self.result, f)
        with open(f"{out}/job.json") as f:
            self.job = json.load(f)

        class Done:
            returncode = self.returncode
            stdout = self.logs

        return Done()


def _runner(tmp_path, run) -> DockerRunner:
    return DockerRunner(
        cache_dir=tmp_path / "cache",
        env={"AWS_PROFILE": "leaf-al-worker", "MLFLOW_TRACKING_URI": "http://localhost:5000"},
        aws_dir=tmp_path / ".aws",
        run=run,
    )


def test_the_job_runs_in_the_image_it_names(tmp_path):
    from common.jobs import parse_job

    run = FakeRun(result={"onnx_sha256": "ab" * 32})
    job = parse_job(json.dumps(_job()))
    assert _runner(tmp_path, run).run(job) == {"onnx_sha256": "ab" * 32}
    (cmd,) = run.commands
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "all"
    # MLflow is reached through the SSM tunnel on the host's localhost:5000.
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "host"
    assert f"{tmp_path / 'cache'}:/cache" in cmd
    assert f"{tmp_path / '.aws'}:/root/.aws:ro" in cmd
    assert "AWS_PROFILE=leaf-al-worker" in cmd
    assert cmd[-4:] == [IMAGE, "python", "-m", "worker.run_job"]
    # The task token never enters the container: only the agent reports.
    assert "task_token" not in run.job and run.job["job_id"] == str(job.job_id)


def test_a_container_that_fails_is_a_failed_job(tmp_path):
    from common.jobs import parse_job

    run = FakeRun(returncode=1, logs="line\n" * 500 + "CUDA out of memory")
    with pytest.raises(JobFailed, match="CUDA out of memory"):
        _runner(tmp_path, run).run(parse_job(json.dumps(_job())))


def test_the_job_output_is_read_as_utf8_whatever_the_host_code_page(tmp_path):
    from common.jobs import parse_job

    run = FakeRun(result={})
    _runner(tmp_path, run).run(parse_job(json.dumps(_job())))
    assert run.kw["encoding"] == "utf-8" and run.kw["errors"] == "replace"


def test_a_container_without_a_result_is_a_failed_job(tmp_path):
    from common.jobs import parse_job

    with pytest.raises(JobFailed, match="no result"):
        _runner(tmp_path, FakeRun()).run(parse_job(json.dumps(_job())))
