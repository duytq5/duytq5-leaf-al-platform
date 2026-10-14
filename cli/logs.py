"""`al logs`: every /leaf-al/ log line about one id, oldest first.

The id is anything a log line names: a round_id, a job id, a release's
dataset or commit, an API request id. A round's state machine lines name the
round only in step inputs, so a first query finds the executions that mention
the id and the second one also takes every line of those executions (a failed
step's error is in them).

`--errors` lists recent errors across everything instead. Runs with your own
AWS credentials (AWS_PROFILE=leaf-al), like the rest of the operator CLI.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import timedelta

PREFIX = "/leaf-al/"
_PLAIN = re.compile(r"^[A-Za-z0-9._:/@+=-]+$")
_SINCE = re.compile(r"^(\d+)([mhd])$")
_UNITS = {"m": "minutes", "h": "hours", "d": "days"}

ERRORS = (
    'filter level = "ERROR" or ispresent(errorType)'
    " or @message like /Failed|TimedOut|Aborted|Traceback/"
)


def client():
    import boto3

    return boto3.client("logs")


def parse_since(text: str) -> timedelta:
    m = _SINCE.match(text)
    if not m:
        raise ValueError(f"{text!r}: use a number and m, h or d, like 30m or 2d")
    return timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})


def log_groups(logs) -> list[str]:
    pages = logs.get_paginator("describe_log_groups").paginate(logGroupNamePrefix=PREFIX)
    return [g["logGroupName"] for page in pages for g in page["logGroups"]]


def run_query(logs, groups, query, start, end, sleep: Callable | None = None) -> list[dict]:
    query_id = logs.start_query(
        logGroupNames=groups,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=query,
    )["queryId"]
    while True:
        answer = logs.get_query_results(queryId=query_id)
        if answer["status"] == "Complete":
            return [{f["field"]: f["value"] for f in row} for row in answer["results"]]
        if answer["status"] not in ("Scheduled", "Running"):
            raise RuntimeError(f"the Logs Insights query ended {answer['status']}")
        (sleep or time.sleep)(1)


def follow(logs, needle, start, end, sleep: Callable | None = None) -> list[dict]:
    if not _PLAIN.match(needle):
        raise ValueError(f"{needle!r} is not an id")
    groups = log_groups(logs)
    executions = [
        row["execution_arn"]
        for row in run_query(
            logs,
            groups,
            f'filter @message like "{needle}" and ispresent(execution_arn)'
            " | stats count(*) by execution_arn",
            start,
            end,
            sleep,
        )
    ]
    match = f'@message like "{needle}"'
    if executions:
        arns = ", ".join(f'"{a}"' for a in executions)
        match += f" or execution_arn in [{arns}]"
    query = (
        f"fields @timestamp, @log, @message | filter {match} | sort @timestamp asc | limit 10000"
    )
    return run_query(logs, groups, query, start, end, sleep)


def recent_errors(logs, start, end, sleep: Callable | None = None) -> list[dict]:
    query = f"fields @timestamp, @log, @message | {ERRORS} | sort @timestamp asc | limit 1000"
    return run_query(logs, log_groups(logs), query, start, end, sleep)


def format_row(row: dict) -> str:
    """`10:00:01  round/select  ERROR failed select_handler round_id=...`"""
    clock = row["@timestamp"].split(" ")[-1].split(".")[0]
    group = row["@log"].split(":", 1)[-1].removeprefix(PREFIX)
    text = row["@message"].strip()
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if isinstance(body, dict):
        text = _describe(body) or text
    return f"{clock}  {group}  {text}"


def _describe(body: dict) -> str:
    if "details" in body or ("type" in body and "execution_arn" in body):  # Step Functions
        details = body.get("details") or {}
        parts = [body.get("type", ""), details.get("name", "")]
        if details.get("error"):
            parts.append(f"{details['error']}: {details.get('cause', '')}")
        return " ".join(p for p in parts if p)
    if "errorType" in body:  # unhandled Lambda error
        return f"{body['errorType']}: {body.get('errorMessage', '')}"
    if "message" in body:  # a Lambda log call
        return f"{body.get('level', '')} {body['message']}".strip()
    return ""
