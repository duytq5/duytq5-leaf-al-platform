import json

import pytest

pytest.importorskip("aws_cdk")

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from infra.app import build_app
from infra.stacks.platform import (
    SSM_DATA_BUCKET,
    SSM_JOB_QUEUE_URL,
    SSM_MODELS_BUCKET,
    PlatformStack,
)

ADMIN = "203.0.113.7/32"
WORKER = "198.51.100.9/32"


def _template(**context) -> Template:
    app = cdk.App(context=context)
    return Template.from_stack(PlatformStack(app, "LeafAlPlatform"))


@pytest.fixture(scope="module")
def full() -> Template:
    return _template(adminCidr=ADMIN, workerCidr=WORKER, budgetEmail="owner@example.com")


@pytest.fixture(scope="module")
def bare() -> Template:
    return _template()


def _statements(template: Template, logical_prefix: str) -> list[dict]:
    policies = template.find_resources("AWS::IAM::Policy")
    (policy,) = [p for k, p in policies.items() if k.startswith(logical_prefix)]
    return policy["Properties"]["PolicyDocument"]["Statement"]


def _actions(stmt: dict) -> list[str]:
    a = stmt["Action"]
    return a if isinstance(a, list) else [a]


def test_app_synthesizes():
    assembly = build_app().synth()
    assert assembly.get_stack_by_name("LeafAlPlatform").template is not None


def test_buckets_private_encrypted_retained(full):
    buckets = full.find_resources("AWS::S3::Bucket")
    assert len(buckets) == 2
    for bucket in buckets.values():
        props = bucket["Properties"]
        assert props["PublicAccessBlockConfiguration"] == {
            "BlockPublicAcls": True,
            "BlockPublicPolicy": True,
            "IgnorePublicAcls": True,
            "RestrictPublicBuckets": True,
        }
        assert "BucketEncryption" in props
        assert bucket["DeletionPolicy"] == "Retain"
        assert bucket["UpdateReplacePolicy"] == "Retain"
    # Both buckets deny plain-HTTP access.
    full.resource_count_is("AWS::S3::BucketPolicy", 2)
    full.all_resources_properties(
        "AWS::S3::BucketPolicy",
        {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Effect": "Deny",
                                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                            }
                        )
                    ]
                )
            }
        },
    )


def test_only_models_bucket_is_versioned(full):
    versioned = [
        b
        for b in full.find_resources("AWS::S3::Bucket").values()
        if b["Properties"].get("VersioningConfiguration", {}).get("Status") == "Enabled"
    ]
    assert len(versioned) == 1


def test_job_queue_has_dlq_and_long_polling(full):
    full.resource_count_is("AWS::SQS::Queue", 2)
    full.has_resource_properties(
        "AWS::SQS::Queue",
        {
            "ReceiveMessageWaitTimeSeconds": 20,
            "VisibilityTimeout": 1800,
            "MessageRetentionPeriod": 14 * 24 * 3600,
            "SqsManagedSseEnabled": True,
            "RedrivePolicy": {
                "deadLetterTargetArn": {"Fn::GetAtt": [Match.string_like_regexp("^JobDlq"), "Arn"]},
                "maxReceiveCount": 3,
            },
        },
    )


def test_ssm_parameter_names(full):
    for name in (SSM_DATA_BUCKET, SSM_MODELS_BUCKET, SSM_JOB_QUEUE_URL):
        full.has_resource_properties("AWS::SSM::Parameter", {"Name": name, "Type": "String"})


def test_worker_policy_is_scoped(full):
    statements = _statements(full, "WorkerUserDefaultPolicy")
    for stmt in statements:
        assert stmt["Effect"] == "Allow"
        for action in _actions(stmt):
            assert not action.endswith("*"), action
        # Only the Step Functions callbacks (no resource-level support) use "*".
        if stmt["Resource"] == "*":
            assert all(a.startswith("states:SendTask") for a in _actions(stmt))
    actions = {a for s in statements for a in _actions(s)}
    assert "s3:DeleteObject" in actions  # MLflow artifacts only
    assert not any(a.startswith(("iam:", "ec2:", "sqs:SendMessage", "sqs:Purge")) for a in actions)

    # Data bucket objects: read raw/ and manifests/, write only scores/.
    writes = [s for s in statements if "s3:PutObject" in _actions(s)]
    written = json.dumps([s["Resource"] for s in writes])
    assert "/scores/*" in written and "/mlflow-artifacts/*" in written
    assert "/raw/*" not in written and "/manifests/*" not in written


def test_no_access_key_in_template(full):
    full.resource_count_is("AWS::IAM::AccessKey", 0)
    full.resource_count_is("AWS::IAM::User", 1)


def test_security_group_only_admin_and_worker(full):
    (sg,) = full.find_resources("AWS::EC2::SecurityGroup").values()
    ingress = sg["Properties"]["SecurityGroupIngress"]
    assert {(r["CidrIp"], r["FromPort"]) for r in ingress} == {
        (ADMIN, 5000),
        (ADMIN, 5432),
        (WORKER, 5000),
        (WORKER, 5432),
    }
    assert all(r["FromPort"] == r["ToPort"] for r in ingress)


def test_security_group_closed_without_admin_cidr(bare):
    (sg,) = bare.find_resources("AWS::EC2::SecurityGroup").values()
    assert "SecurityGroupIngress" not in sg["Properties"]


@pytest.mark.parametrize("cidr", ["0.0.0.0/0", "not-an-ip", "10.0.0.1/24"])
def test_bad_admin_cidr_rejected(cidr):
    with pytest.raises(ValueError):
        _template(adminCidr=cidr)


def test_data_volume_retained_and_attached(full):
    volumes = full.find_resources("AWS::EC2::Volume")
    (volume,) = volumes.values()
    assert volume["DeletionPolicy"] == "Retain"
    assert volume["Properties"]["Encrypted"] is True
    assert volume["Properties"]["Size"] == 20
    full.has_resource_properties(
        "AWS::EC2::VolumeAttachment",
        {"Device": "/dev/sdf", "VolumeId": {"Ref": next(iter(volumes))}},
    )


def test_instance_no_ssh_imdsv2_ami_resolved_at_launch(full):
    (instance,) = full.find_resources("AWS::EC2::Instance").values()
    props = instance["Properties"]
    assert "KeyName" not in props
    assert props["ImageId"].startswith("resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023")
    assert props["InstanceType"] == "t3.small"
    full.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {"LaunchTemplateData": {"MetadataOptions": {"HttpTokens": "required"}}},
    )
    user_data = json.dumps(props["UserData"])
    assert "mkfs.ext4" in user_data and "blkid" in user_data  # format only blank volumes


def test_no_nat_gateway(full):
    full.resource_count_is("AWS::EC2::NatGateway", 0)


def test_budget_only_with_email(full, bare):
    full.has_resource_properties(
        "AWS::Budgets::Budget",
        {"Budget": {"BudgetLimit": {"Amount": 20, "Unit": "USD"}, "TimeUnit": "MONTHLY"}},
    )
    bare.resource_count_is("AWS::Budgets::Budget", 0)


def test_context_overrides():
    t = _template(instanceType="t3.medium", dataVolumeGb="30", budgetEmail="a@b.c", budgetUsd="35")
    t.has_resource_properties("AWS::EC2::Instance", {"InstanceType": "t3.medium"})
    t.has_resource_properties("AWS::EC2::Volume", {"Size": 30})
    t.has_resource_properties(
        "AWS::Budgets::Budget", {"Budget": {"BudgetLimit": {"Amount": 35, "Unit": "USD"}}}
    )
