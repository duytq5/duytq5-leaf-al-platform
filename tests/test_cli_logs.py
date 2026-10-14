import json
from datetime import UTC, datetime, timedelta

import pytest

from cli import logs
from cli.main import main

ROUND_ID = "0b9e6f9c-6a51-4c7e-9d0f-1f2a3b4c5d6e"
ARN = "arn:aws:states:ap-southeast-2:1:execution:al-round:r1"
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


class FakeLogs:
    """The CloudWatch Logs calls `al logs` makes, with canned answers."""

    def __init__(self, answers: list[list[dict]], groups=None):
        self.answers = answers
        self.groups = groups or ["/leaf-al/api", "/leaf-al/states/al-round"]
        self.queries: list[dict] = []
        self.polls = 0

    def get_paginator(self, name):
        assert name == "describe_log_groups"
        groups = self.groups

        class Pages:
            def paginate(self, logGroupNamePrefix):
                assert logGroupNamePrefix == "/leaf-al/"
                yield {"logGroups": [{"logGroupName": g} for g in groups[:1]]}
                yield {"logGroups": [{"logGroupName": g} for g in groups[1:]]}

        return Pages()

    def start_query(self, **kwargs):
        self.queries.append(kwargs)
        return {"queryId": f"q{len(self.queries)}"}

    def get_query_results(self, queryId):
        self.polls += 1
        if self.polls % 2:  # every query is still running on its first poll
            return {"status": "Running", "results": []}
        rows = self.answers[int(queryId[1:]) - 1]
        return {
            "status": "Complete",
            "results": [[{"field": k, "value": v} for k, v in r.items()] for r in rows],
        }


def _row(ts: str, group: str, message: dict | str) -> dict:
    body = message if isinstance(message, str) else json.dumps(message)
    return {"@timestamp": f"2026-10-14 {ts}.000", "@log": f"123:{group}", "@message": body}


def test_since():
    assert logs.parse_since("30m") == timedelta(minutes=30)
    assert logs.parse_since("6h") == timedelta(hours=6)
    assert logs.parse_since("7d") == timedelta(days=7)
    with pytest.raises(ValueError):
        logs.parse_since("soon")


@pytest.mark.parametrize("needle", ['a"b', "a b", "x|y", ""])
def test_ids_are_plain(needle):
    with pytest.raises(ValueError):
        logs.follow(FakeLogs([]), needle, NOW - timedelta(days=1), NOW, sleep=lambda _: None)


def test_follow_adds_the_executions_that_mention_the_id():
    client = FakeLogs(
        [
            [{"execution_arn": ARN, "count()": "4"}],
            [_row("10:00:01", "/leaf-al/states/al-round", {"type": "ExecutionStarted"})],
        ]
    )
    rows = logs.follow(client, ROUND_ID, NOW - timedelta(days=1), NOW, sleep=lambda _: None)
    first, second = client.queries
    for q in (first, second):
        assert q["logGroupNames"] == ["/leaf-al/api", "/leaf-al/states/al-round"]
        assert q["startTime"] == int((NOW - timedelta(days=1)).timestamp())
        assert q["endTime"] == int(NOW.timestamp())
    assert ROUND_ID in first["queryString"]
    assert "stats" in first["queryString"]
    assert ROUND_ID in second["queryString"]
    assert ARN in second["queryString"]
    assert "sort @timestamp asc" in second["queryString"]
    assert len(rows) == 1


def test_follow_without_executions_filters_on_the_id_only():
    client = FakeLogs([[], []])
    logs.follow(client, ROUND_ID, NOW - timedelta(days=1), NOW, sleep=lambda _: None)
    assert "execution_arn in" not in client.queries[1]["queryString"]


def test_recent_errors_is_one_query():
    client = FakeLogs([[_row("10:00:00", "/leaf-al/api", "Traceback")]])
    rows = logs.recent_errors(client, NOW - timedelta(hours=1), NOW, sleep=lambda _: None)
    (q,) = client.queries
    assert "ERROR" in q["queryString"] and "Failed" in q["queryString"]
    assert len(rows) == 1


def test_failed_query_raises():
    class Broken(FakeLogs):
        def get_query_results(self, queryId):
            return {"status": "Failed", "results": []}

    with pytest.raises(RuntimeError):
        logs.recent_errors(Broken([]), NOW, NOW, sleep=lambda _: None)


def test_format_lambda_line():
    line = logs.format_row(
        _row(
            "10:00:01",
            "/leaf-al/round/select",
            {"level": "ERROR", "message": f"failed select_handler round_id={ROUND_ID}"},
        )
    )
    assert line.startswith("10:00:01  round/select")
    assert "ERROR" in line and "failed select_handler" in line


def test_format_unhandled_lambda_error():
    line = logs.format_row(
        _row(
            "10:00:01",
            "/leaf-al/round/select",
            {"errorType": "RoundError", "errorMessage": "no scores", "stackTrace": ["..."]},
        )
    )
    assert "RoundError: no scores" in line


def test_format_state_machine_event():
    line = logs.format_row(
        _row(
            "10:00:02",
            "/leaf-al/states/al-round",
            {
                "type": "TaskFailed",
                "execution_arn": ARN,
                "details": {"error": "RoundError", "cause": "no scores"},
            },
        )
    )
    assert "states/al-round" in line
    assert "TaskFailed" in line and "RoundError" in line and "no scores" in line


def test_format_state_entered_names_the_step():
    line = logs.format_row(
        _row(
            "10:00:02",
            "/leaf-al/states/al-round",
            {"type": "TaskStateEntered", "details": {"name": "Select", "input": "{}"}},
        )
    )
    assert "TaskStateEntered Select" in line


def test_format_plain_text():
    assert logs.format_row(_row("10:00:03", "/leaf-al/api", "hello")).endswith("api  hello")


def test_cli_prints_one_line_per_row(monkeypatch, capsys):
    client = FakeLogs(
        [
            [],
            [
                _row("10:00:01", "/leaf-al/api", "a"),
                _row("10:00:02", "/leaf-al/api", "b"),
            ],
        ]
    )
    monkeypatch.setattr(logs, "client", lambda: client)
    monkeypatch.setattr(logs.time, "sleep", lambda _: None)
    assert main(["logs", ROUND_ID, "--since", "2h"]) == 0
    assert capsys.readouterr().out.splitlines() == ["10:00:01  api  a", "10:00:02  api  b"]


def test_cli_errors(monkeypatch, capsys):
    client = FakeLogs([[]])
    monkeypatch.setattr(logs, "client", lambda: client)
    monkeypatch.setattr(logs.time, "sleep", lambda _: None)
    assert main(["logs", "--errors"]) == 0
    assert "no log lines" in capsys.readouterr().err


def test_cli_needs_an_id_or_errors(capsys):
    with pytest.raises(SystemExit):
        main(["logs"])
