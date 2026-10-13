"""Upload validation wiring: S3 incoming/ -> SQS -> UploadFn (in the VPC)."""

import json

import pytest

pytest.importorskip("aws_cdk")

import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.lambda_code import UPLOAD
from infra.stacks.platform import PlatformStack


@pytest.fixture(scope="module")
def t() -> Template:
    return Template.from_stack(PlatformStack(cdk.App(), "LeafAlPlatform"))


def _fn(t: Template) -> tuple[str, dict]:
    ((logical_id, fn),) = [
        (k, f)
        for k, f in t.find_resources("AWS::Lambda::Function").items()
        if f["Properties"].get("Handler") == "lambdas.upload.handler"
    ]
    return logical_id, fn["Properties"]


def _queues(t: Template) -> dict[str, dict]:
    return {
        k: q["Properties"]
        for k, q in t.find_resources("AWS::SQS::Queue").items()
        if k.startswith("Upload")
    }


def test_upload_lambda_runs_in_the_vpc_with_postgres(t):
    _, fn = _fn(t)
    assert fn["Runtime"] == "python3.12"
    assert fn["VpcConfig"]["SubnetIds"] == [{"Ref": "LambdaSubnet"}]
    env = fn["Environment"]["Variables"]
    assert env["PG_HOSTADDR"] == {"Fn::GetAtt": ["HostB4E45AD7", "PrivateIp"]}
    assert "{{resolve:secretsmanager:leaf-al/postgres-password" in json.dumps(env["PG_PASSWORD"])
    assert env["DATA_BUCKET"]["Ref"].startswith("DataBucket")
    assert fn["MemorySize"] >= 1024  # a 40-megapixel photo decoded is ~120 MB


def test_incoming_uploads_notify_the_queue(t):
    (notifications,) = t.find_resources("Custom::S3BucketNotifications").values()
    props = notifications["Properties"]
    assert props["BucketName"]["Ref"].startswith("DataBucket")
    (config,) = props["NotificationConfiguration"]["QueueConfigurations"]
    assert config["Events"] == ["s3:ObjectCreated:*"]
    rules = config["Filter"]["Key"]["FilterRules"]
    assert rules == [{"Name": "prefix", "Value": "incoming/"}]
    assert config["QueueArn"]["Fn::GetAtt"][0].startswith("UploadQueue")


def test_queue_keeps_uploads_while_the_host_is_stopped(t):
    queues = _queues(t)
    (main_id,) = [k for k in queues if k.startswith("UploadQueue")]
    (dlq_id,) = [k for k in queues if k.startswith("UploadDlq")]
    main, dlq = queues[main_id], queues[dlq_id]
    _, fn = _fn(t)
    assert main["VisibilityTimeout"] >= fn["Timeout"]
    redrive = main["RedrivePolicy"]
    assert redrive["deadLetterTargetArn"]["Fn::GetAtt"][0] == dlq_id
    # Retried for about a day before the DLQ, which keeps them two weeks.
    assert main["VisibilityTimeout"] * redrive["maxReceiveCount"] >= 24 * 3600
    assert dlq["MessageRetentionPeriod"] == 14 * 24 * 3600
    assert main["SqsManagedSseEnabled"] is True


def test_lambda_reads_the_queue_and_reports_failed_messages(t):
    fn_id, _ = _fn(t)
    (mapping,) = [
        m["Properties"]
        for m in t.find_resources("AWS::Lambda::EventSourceMapping").values()
        if m["Properties"]["FunctionName"] == {"Ref": fn_id}
    ]
    assert mapping["EventSourceArn"]["Fn::GetAtt"][0].startswith("UploadQueue")
    assert mapping["FunctionResponseTypes"] == ["ReportBatchItemFailures"]


def test_lambda_may_only_move_uploads(t):
    _, fn = _fn(t)
    role = fn["Role"]["Fn::GetAtt"][0]
    s3 = {}
    for policy in t.find_resources("AWS::IAM::Policy").values():
        if {"Ref": role} not in policy["Properties"]["Roles"]:
            continue
        for st in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            for action in actions:
                if action.startswith("s3:"):
                    s3[action] = json.dumps(st["Resource"])
    assert set(s3) == {"s3:GetObject", "s3:DeleteObject", "s3:PutObject"}
    assert "/incoming/*" in s3["s3:GetObject"] and "/incoming/*" in s3["s3:DeleteObject"]
    assert "/raw/*" in s3["s3:PutObject"]
    assert all("DataBucket" in r for r in s3.values())


def test_unprocessed_uploads_expire(t):
    (bucket,) = [
        b["Properties"]
        for k, b in t.find_resources("AWS::S3::Bucket").items()
        if k.startswith("DataBucket")
    ]
    rules = {r.get("Prefix"): r for r in bucket["LifecycleConfiguration"]["Rules"] if "Prefix" in r}
    assert rules["incoming/"]["ExpirationInDays"] == 14


def test_upload_bundle_is_small():
    names = {r.split("==")[0] for r in UPLOAD.requirements()}
    assert {"pillow", "psycopg", "psycopg-binary"} <= names
    assert not {"numpy", "pyarrow", "fastapi"} & names
