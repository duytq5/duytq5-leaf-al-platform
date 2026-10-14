"""The worker agent on the GPU machine (design doc "Worker").

Long-polls the job queue. For each job it runs the worker image the job names
(built by GitHub Actions for the commit that started the round or trained the
model) with the GPU, then reports to Step Functions with the job's task token:
SendTaskSuccess with the job's result, or SendTaskFailure. The task token
stays with the agent; the container only gets the job.

  * While a job runs, its message is kept hidden (ChangeMessageVisibility), so
    SQS does not hand it out twice. If the agent dies, SQS hands it out again
    once the visibility timeout ends and the job runs again; jobs are
    idempotent on job_id.
  * A message is deleted once Step Functions has the outcome, or when the
    state machine no longer waits for it (timed out, stopped).
  * A report that does not reach Step Functions leaves the message in the
    queue, to be run and reported again later.

Run on the GPU machine (needs Docker with the NVIDIA runtime, the
`leaf-al-worker` AWS profile and the MLflow tunnel):

    AWS_PROFILE=leaf-al-worker MLFLOW_TRACKING_URI=http://localhost:5000 \\
        uv run python -m worker.agent
"""

import json
import logging
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from pydantic import ValidationError

from common.jobs import parse_job

log = logging.getLogger("worker.agent")

# Step Functions caps a failure cause at 32,768 characters.
MAX_CAUSE = 32_768
# The state machine stopped waiting for this task: nobody wants the result.
GONE = {"TaskTimedOut", "TaskDoesNotExist", "InvalidToken"}
# Matches the job queue's visibility timeout (infra JOB_VISIBILITY_TIMEOUT).
VISIBILITY_TIMEOUT = 30 * 60
SSM_JOB_QUEUE_URL = "/leaf-al/queues/jobs-url"


class JobFailed(RuntimeError):
    """The job ran and failed; the state machine is told and fails the round
    or release."""


class Agent:
    def __init__(
        self,
        *,
        sqs,
        sfn,
        queue_url: str,
        runner: Callable[[Any], dict],
        visibility_timeout: int = VISIBILITY_TIMEOUT,
        extend_every: float = VISIBILITY_TIMEOUT / 3,
    ) -> None:
        self.sqs, self.sfn, self.queue_url, self.runner = sqs, sfn, queue_url, runner
        self.visibility_timeout = visibility_timeout
        self.extend_every = extend_every

    def poll_once(self) -> int:
        """Wait up to 20 s for a job and handle it; returns messages handled."""
        response = self.sqs.receive_message(
            QueueUrl=self.queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=20
        )
        messages = response.get("Messages", [])
        for message in messages:
            self._handle(message)
        return len(messages)

    def run_forever(self) -> None:
        while True:
            self.poll_once()

    def _handle(self, message: dict) -> None:
        receipt = message["ReceiptHandle"]
        try:
            job = parse_job(message["Body"])
        except ValidationError as e:
            log.error("message %s is not a valid job: %s", message["MessageId"], e)
            token = _token(message["Body"])
            if token is None or self._report(token, failure=("InvalidJob", str(e))):
                self._delete(receipt)
            return

        log.info("job %s (%s) in %s", job.job_id, job.type, job.image)
        stop = threading.Event()
        keeper = threading.Thread(target=self._keep_hidden, args=(receipt, stop), daemon=True)
        keeper.start()
        try:
            result = self.runner(job)
            outcome = {"success": result}
        except Exception as e:  # any failure of the job fails its task
            log.exception("job %s failed", job.job_id)
            outcome = {"failure": (type(e).__name__, str(e))}
        finally:
            stop.set()
            keeper.join()
        if self._report(job.task_token, **outcome):
            self._delete(receipt)

    def _report(self, token: str, success: dict | None = None, failure=None) -> bool:
        """Tell Step Functions; True when the message can be deleted."""
        try:
            if failure is None:
                self.sfn.send_task_success(taskToken=token, output=json.dumps(success))
            else:
                error, cause = failure
                self.sfn.send_task_failure(
                    taskToken=token, error=error[:256], cause=cause[-MAX_CAUSE:]
                )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in GONE:
                log.warning("the state machine no longer waits for this job (%s)", code)
                return True
            log.error("could not report the job (%s); it will run again", code)
            return False
        return True

    def _delete(self, receipt: str) -> None:
        self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt)

    def _keep_hidden(self, receipt: str, stop: threading.Event) -> None:
        while not stop.wait(self.extend_every):
            try:
                self.sqs.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt,
                    VisibilityTimeout=self.visibility_timeout,
                )
            except ClientError as e:
                log.warning("could not extend the job's visibility: %s", e)


def _token(body: str) -> str | None:
    try:
        token = json.loads(body).get("task_token")
    except (ValueError, AttributeError):
        return None
    return token if isinstance(token, str) and token else None


class DockerRunner:
    """Runs a job in its worker image, with the GPU, the image cache and the
    worker's AWS profile, on the host network (MLflow tunnel on localhost)."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        env: dict[str, str],
        aws_dir: Path,
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.cache_dir, self.env, self.aws_dir, self.run_process = cache_dir, env, aws_dir, run

    def __call__(self, job) -> dict:
        return self.run(job)

    def run(self, job) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="leaf-al-job-") as out:
            (Path(out) / "job.json").write_text(job.model_dump_json(exclude={"task_token"}))
            cmd = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--network",
                "host",
                "--shm-size",
                "8g",
            ]
            cmd += ["-v", f"{self.cache_dir}:/cache", "-v", f"{out}:/out"]
            cmd += ["-v", f"{self.aws_dir}:/root/.aws:ro"]
            for key, value in sorted(self.env.items()):
                cmd += ["-e", f"{key}={value}"]
            cmd += [job.image, "python", "-m", "worker.run_job"]
            done = self.run_process(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
            )
            logs = done.stdout or ""
            print(logs, end="", flush=True)
            if done.returncode != 0:
                raise JobFailed(f"the job exited with {done.returncode}:\n{logs[-4000:]}")
            result = Path(out) / "result.json"
            if not result.is_file():
                raise JobFailed(f"the job wrote no result:\n{logs[-4000:]}")
            return json.loads(result.read_text())


# Settings the container gets from the agent's environment.
PASSED_ENV = (
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "MLFLOW_TRACKING_URI",
    "MODELS_BUCKET",  # optional; the job reads it from SSM otherwise
    "WORKER_NUM_WORKERS",
)


def main() -> None:  # pragma: no cover - wiring only
    import boto3

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    session = boto3.Session()
    queue_url = (
        os.environ.get("JOB_QUEUE_URL")
        or session.client("ssm").get_parameter(Name=SSM_JOB_QUEUE_URL)["Parameter"]["Value"]
    )
    runner = DockerRunner(
        cache_dir=Path(os.environ.get("WORKER_CACHE_DIR", "~/.cache/leaf-al")).expanduser(),
        env={k: os.environ[k] for k in PASSED_ENV if k in os.environ},
        aws_dir=Path("~/.aws").expanduser(),
    )
    agent = Agent(
        sqs=session.client("sqs"),
        sfn=session.client("stepfunctions"),
        queue_url=queue_url,
        runner=runner,
    )
    log.info("waiting for jobs on %s", queue_url)
    agent.run_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
