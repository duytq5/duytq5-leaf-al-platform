"""The single CDK stack for the POC.

Phase 1 adds: S3 buckets (al-data, al-models), SQS job queue + DLQ, the
al-round state machine, select/oracle Lambdas, IAM, and a Budget alert.
"""

from aws_cdk import Stack, Tags
from constructs import Construct


class PlatformStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        Tags.of(self).add("project", "leaf-al-platform")
