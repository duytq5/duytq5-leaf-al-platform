"""Log groups: everything under /leaf-al/, so one search covers the whole flow.

`al logs <round_id>` and the saved CloudWatch queries read every group with
this prefix. Lambdas log JSON: `extra` fields of a log call (round_id, ...)
become fields Logs Insights can filter on.
"""

from aws_cdk import RemovalPolicy
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

PREFIX = "/leaf-al/"
RETENTION = logs.RetentionDays.ONE_MONTH


def log_group(scope: Construct, construct_id: str, name: str) -> logs.LogGroup:
    """/leaf-al/{name}; dropped with the stack (debug logs, not records)."""
    return logs.LogGroup(
        scope,
        construct_id,
        log_group_name=PREFIX + name,
        retention=RETENTION,
        removal_policy=RemovalPolicy.DESTROY,
    )


def lambda_logging(scope: Construct, construct_id: str, name: str) -> dict:
    """Keyword arguments for lambda_.Function: its own group, JSON lines."""
    return {
        "log_group": log_group(scope, construct_id, name),
        "logging_format": lambda_.LoggingFormat.JSON,
    }


def kebab(construct_name: str) -> str:
    """MarkRunningFn -> mark-running."""
    name = construct_name.removesuffix("Fn")
    return "".join(f"-{c.lower()}" if c.isupper() else c for c in name).lstrip("-")


# Saved in the CloudWatch console (Logs Insights > Queries). Replace ROUND_ID
# with a round id, a release's dataset or commit, a job id or a request id.
FOLLOW_QUERY = """fields @timestamp, @log, @message
| filter @message like "ROUND_ID"
| sort @timestamp asc
| limit 10000"""

ERRORS_QUERY = """fields @timestamp, @log, @message
| filter level = "ERROR" or ispresent(errorType)
    or @message like /Failed|TimedOut|Aborted|Traceback/
| sort @timestamp desc
| limit 1000"""


def saved_queries(scope: Construct) -> None:
    """The two console queries, over every log group in `scope`."""
    groups = [c.log_group_name for c in scope.node.find_all() if isinstance(c, logs.LogGroup)]
    for construct_id, name, query in (
        ("FollowQuery", "leaf-al/follow-an-id", FOLLOW_QUERY),
        ("ErrorsQuery", "leaf-al/recent-errors", ERRORS_QUERY),
    ):
        logs.CfnQueryDefinition(
            scope, construct_id, name=name, query_string=query, log_group_names=groups
        )
