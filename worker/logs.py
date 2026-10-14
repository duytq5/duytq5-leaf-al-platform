"""The worker's logs in CloudWatch (/leaf-al/worker), next to the Lambdas'.

Each line is JSON like the Lambdas' (level, logger, message) plus the job it
belongs to (job_id, round_id, job_type), so `al logs <round_id>` shows the
worker's part of a round. Lines are buffered and sent every few seconds by a
background thread. Logging never stops the worker: when CloudWatch cannot be
reached, lines are dropped (they still reach the terminal) and it says so once.
"""

import json
import logging
import sys
import threading
import time
from contextvars import ContextVar

from botocore.exceptions import BotoCoreError, ClientError

LOG_GROUP = "/leaf-al/worker"

# The job the agent is running; empty between jobs.
job_context: ContextVar[dict | None] = ContextVar("job_context", default=None)


def job_fields(job) -> dict[str, str]:
    fields = {"job_id": str(job.job_id), "job_type": str(job.type)}
    if job.round_id is not None:
        fields["round_id"] = str(job.round_id)
    if getattr(job, "training_id", None) is not None:
        fields["training_id"] = str(job.training_id)
    return fields


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **(job_context.get() or {}),
        }
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line)


class CloudWatchHandler(logging.Handler):
    # PutLogEvents takes at most 10,000 events and 1 MB per call, 256 KB per
    # event (26 bytes of overhead each).
    MAX_EVENTS = 1000
    MAX_BATCH_BYTES = 900_000
    MAX_EVENT_BYTES = 200_000

    def __init__(
        self, client, group: str, stream: str, *, flush_every: float = 5.0, max_events=None
    ) -> None:
        super().__init__()
        self.client, self.group, self.stream = client, group, stream
        self.flush_every = flush_every
        self.max_events = max_events or self.MAX_EVENTS
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._warned = False

    def open(self) -> "CloudWatchHandler":
        try:
            self.client.create_log_stream(logGroupName=self.group, logStreamName=self.stream)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                self._warn(e)
        except BotoCoreError as e:
            self._warn(e)
        self._thread.start()
        return self

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        data = message.encode()
        if len(data) > self.MAX_EVENT_BYTES:
            message = data[: self.MAX_EVENT_BYTES].decode(errors="ignore")
        with self._lock:
            self._buffer.append({"timestamp": int(record.created * 1000), "message": message})

    def flush(self) -> None:
        with self._lock:
            events, self._buffer = self._buffer, []
        events.sort(key=lambda e: e["timestamp"])
        batch: list[dict] = []
        size = 0
        for event in events:
            event_size = len(event["message"].encode()) + 26
            if batch and (
                len(batch) >= self.max_events or size + event_size > self.MAX_BATCH_BYTES
            ):
                self._send(batch)
                batch, size = [], 0
            batch.append(event)
            size += event_size
        if batch:
            self._send(batch)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self.flush()
        super().close()

    def _loop(self) -> None:
        while not self._stop.wait(self.flush_every):
            self.flush()

    def _send(self, batch: list[dict]) -> None:
        try:
            self.client.put_log_events(
                logGroupName=self.group, logStreamName=self.stream, logEvents=batch
            )
        except (ClientError, BotoCoreError) as e:
            self._warn(e)

    def _warn(self, error: Exception) -> None:
        if not self._warned:
            self._warned = True
            print(f"worker logs are not reaching CloudWatch: {error}", file=sys.stderr)


def to_cloudwatch(client, stream: str, group: str = LOG_GROUP) -> CloudWatchHandler:
    """Send every log line of this process to CloudWatch as well."""
    handler = CloudWatchHandler(client, group, stream)
    handler.setFormatter(JsonFormatter())
    logging.getLogger().addHandler(handler.open())
    return handler


def stream_name(host: str, now: float | None = None) -> str:
    """One stream per agent run: gpu-box/2026-10-14T09-30-00."""
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(now))
    return f"{host}/{stamp}"
