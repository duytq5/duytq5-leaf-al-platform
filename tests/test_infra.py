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

# Tests check the template, not the Lambda code asset: skip bundling it.
NO_BUNDLING = {"aws:cdk:bundling-stacks": []}


def _template(**context) -> Template:
    app = cdk.App(context={**NO_BUNDLING, **context})
    return Template.from_stack(PlatformStack(app, "LeafAlPlatform"))


@pytest.fixture(scope="module")
def full() -> Template:
    return _template(budgetEmail="owner@example.com")


@pytest.fixture(scope="module")
def bare() -> Template:
    return _template()


def _statements(template: Template, logical_prefix: str) -> list[dict]:
    policies = {
        **template.find_resources("AWS::IAM::Policy"),
        **template.find_resources("AWS::IAM::ManagedPolicy"),
    }
    (policy,) = [p for k, p in policies.items() if k.startswith(logical_prefix)]
    return policy["Properties"]["PolicyDocument"]["Statement"]


def _actions(stmt: dict) -> list[str]:
    a = stmt["Action"]
    return a if isinstance(a, list) else [a]


def test_app_synthesizes():
    assembly = build_app(NO_BUNDLING).synth()
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
    # The job queue and its DLQ, and the upload queue and its DLQ (tests/test_upload_infra.py).
    full.resource_count_is("AWS::SQS::Queue", 4)
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
    statements = _statements(full, "WorkerPolicy")
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


def test_worker_may_publish_exported_models(full):
    """The export job (F5) checks for edge/{model}/{version}/manifest.json and
    writes model.onnx and manifest.json; it never deletes there."""
    statements = _statements(full, "WorkerPolicy")
    (edge,) = [s for s in statements if s.get("Sid") == "PublishModels"]
    assert sorted(_actions(edge)) == ["s3:GetObject", "s3:PutObject"]
    assert "ModelsBucket" in json.dumps(edge["Resource"]) and "/edge/*" in json.dumps(
        edge["Resource"]
    )
    (listing,) = [s for s in statements if s.get("Sid") == "ListMlflowArtifacts"]
    assert "edge/*" in listing["Condition"]["StringLike"]["s3:prefix"]


def test_worker_policy_fits_iam_limits(full):
    # IAM caps a user's inline policies at 2,048 characters, which the worker
    # outgrew (CREATE_FAILED on the first real deploy). Managed policies allow
    # 6,144 characters, not counting whitespace.
    user_policies = [
        p for p in full.find_resources("AWS::IAM::Policy").values() if p["Properties"].get("Users")
    ]
    assert user_policies == []
    (managed,) = full.find_resources("AWS::IAM::ManagedPolicy").values()
    ((user_ref,),) = [[u["Ref"] for u in managed["Properties"]["Users"]]]
    assert user_ref.startswith("WorkerUser")
    size = len(json.dumps(managed["Properties"]["PolicyDocument"], separators=(",", ":")))
    assert size < 6144 * 0.8, size  # rendered tokens count too; keep headroom


def test_no_access_key_in_template(full):
    full.resource_count_is("AWS::IAM::AccessKey", 0)
    full.resource_count_is("AWS::IAM::User", 1)


def _security_groups(template: Template) -> dict[str, dict]:
    return {
        k.split("Sg")[0]: v for k, v in template.find_resources("AWS::EC2::SecurityGroup").items()
    }


def test_host_takes_postgres_only_from_the_lambdas(full):
    groups = _security_groups(full)
    assert set(groups) == {"Host", "Lambda"}
    assert "SecurityGroupIngress" not in groups["Lambda"]["Properties"]
    # From the internet only Label Studio's HTTPS entry (tests/test_label_studio_infra.py).
    public = groups["Host"]["Properties"]["SecurityGroupIngress"]
    assert sorted(r["FromPort"] for r in public) == [80, 443]
    rules = [
        r["Properties"] for r in full.find_resources("AWS::EC2::SecurityGroupIngress").values()
    ]
    (props,) = [r for r in rules if r["FromPort"] == 5432]
    assert (props["IpProtocol"], props["FromPort"], props["ToPort"]) == ("tcp", 5432, 5432)
    assert props["GroupId"]["Fn::GetAtt"][0].startswith("HostSg")
    assert props["SourceSecurityGroupId"]["Fn::GetAtt"][0].startswith("LambdaSg")
    assert all("CidrIp" not in r for r in rules)


def test_old_cidr_context_is_ignored():
    template = _template(adminCidr="203.0.113.7/32", workerCidr="198.51.100.9/32")
    (host,) = [
        g for k, g in template.find_resources("AWS::EC2::SecurityGroup").items()
        if k.startswith("HostSg")
    ]  # fmt: skip
    cidrs = {r["CidrIp"] for r in host["Properties"]["SecurityGroupIngress"]}
    assert cidrs == {"0.0.0.0/0"}  # Label Studio's 80/443, not the old CIDRs
    rules = template.find_resources("AWS::EC2::SecurityGroupIngress").values()
    assert not any("CidrIp" in r["Properties"] for r in rules)


def test_worker_can_only_port_forward_to_host(full):
    statements = _statements(full, "WorkerPolicy")
    (tunnel,) = [s for s in statements if s.get("Sid") == "TunnelToHost"]
    assert _actions(tunnel) == ["ssm:StartSession"]
    assert tunnel["Condition"] == {"BoolIfExists": {"ssm:SessionDocumentAccessCheck": "true"}}
    resources = json.dumps(tunnel["Resource"])
    assert ':instance/", {"Ref": "Host' in resources  # this one instance only
    assert "document/AWS-StartPortForwardingSession" in resources
    assert "SSM-SessionManagerRunShell" not in resources
    (own,) = [s for s in statements if s.get("Sid") == "ManageOwnSessions"]
    assert "session/${aws:username}-*" in json.dumps(own["Resource"])
    # Shell commands on the host stay with the owner.
    actions = {a for s in statements for a in _actions(s)}
    assert "ssm:SendCommand" not in actions


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
    assert props["InstanceType"] == "t3.medium"
    full.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {"LaunchTemplateData": {"MetadataOptions": {"HttpTokens": "required"}}},
    )
    user_data = json.dumps(props["UserData"])
    assert "mkfs.ext4" in user_data and "blkid" in user_data  # format only blank volumes
    assert "mkswap /swapfile" in user_data
    # First boot also brings up the app, after /data is mounted.
    assert user_data.index("mount -a") < user_data.index("deploy/ec2/bootstrap.sh")
    assert "git clone https://github.com/duytq5/duytq5-leaf-al-platform.git" in user_data
    assert "AWS_REGION=" in user_data and '{"Ref": "AWS::Region"}' in user_data


def test_no_nat_gateway(full):
    full.resource_count_is("AWS::EC2::NatGateway", 0)
    # The one Elastic IP is the host's (Label Studio), not a NAT gateway's.
    full.resource_count_is("AWS::EC2::EIP", 1)
    # The only route is the host subnet's internet route: Lambdas get none.
    (route,) = full.find_resources("AWS::EC2::Route").values()
    assert "LambdaRouteTable" not in json.dumps(route)


def test_lambda_subnet_leaves_the_host_subnet_alone(full):
    subnets = full.find_resources("AWS::EC2::Subnet")
    # The host's subnet keeps its logical id and range, so it is not replaced.
    assert subnets["VpcpublicSubnet1Subnet2BB74ED7"]["Properties"]["CidrBlock"] == "10.0.0.0/16"
    lambdas = subnets["LambdaSubnet"]["Properties"]
    assert lambdas["CidrBlock"] == "10.1.0.0/24"
    assert lambdas["MapPublicIpOnLaunch"] is False
    assert "Host" + "B4E45AD7" in full.find_resources("AWS::EC2::Instance")
    full.has_resource_properties(
        "AWS::EC2::VPCEndpoint",
        {"VpcEndpointType": "Gateway", "RouteTableIds": [{"Ref": "LambdaRouteTable"}]},
    )


def _round_functions(template: Template) -> dict[str, dict]:
    return {
        k: v["Properties"]
        for k, v in template.find_resources("AWS::Lambda::Function").items()
        if k.startswith("AlRound")
    }


def test_round_lambdas_run_in_the_lambda_subnet(full):
    functions = _round_functions(full)
    assert {p["Handler"] for p in functions.values()} == {
        *(
            f"lambdas.round.{h}_handler"
            for h in ("mark_running", "snapshot", "select", "merge", "fail")
        ),
        "lambdas.labeling.start_handler",
        "lambdas.labeling.record_handler",
    }
    for props in functions.values():
        assert props["Runtime"] == "python3.12"
        assert props["VpcConfig"]["SubnetIds"] == [{"Ref": "LambdaSubnet"}]
        env = props["Environment"]["Variables"]
        assert env["PG_HOSTADDR"] == {"Fn::GetAtt": ["HostB4E45AD7", "PrivateIp"]}
        # Resolved by CloudFormation at deploy time: never in the template.
        assert "{{resolve:secretsmanager:leaf-al/postgres-password" in json.dumps(
            env["PG_PASSWORD"]
        )
        assert env["PG_CA"]["Ref"].startswith("SsmParameterValueleafalpostgrestlsca")
        assert "DATABASE_URL" not in env


def _definition(template: Template, name: str = "al-round") -> dict:
    (machine,) = [
        m
        for m in template.find_resources("AWS::StepFunctions::StateMachine").values()
        if m["Properties"]["StateMachineName"] == name
    ]
    parts = machine["Properties"]["DefinitionString"]["Fn::Join"][1]
    return json.loads("".join(p if isinstance(p, str) else "ref" for p in parts))


def _order(definition: dict) -> list[str]:
    states, order, name = definition["States"], [], definition["StartAt"]
    while name in states and states[name].get("Next"):
        order.append(name)
        name = states[name]["Next"]
    return [*order, name]


def test_round_state_machine_runs_the_f4_steps_in_order(full):
    definition = _definition(full)
    states = definition["States"]
    # A round that trains (the default branch of the Choice).
    assert _order(definition)[:3] == ["MarkRunning", "Snapshot", "TrainOrNot"]
    trains = _order({**definition, "StartAt": states["TrainOrNot"]["Default"]})
    assert trains == [
        "Train", "UseTrainedModel", "Score", "Select", "AwaitLabels", "Merge", "RoundSucceeded",
    ]  # fmt: skip
    for step in ["MarkRunning", "Snapshot", "Train", "Score", "Select", "AwaitLabels", "Merge"]:
        assert states[step]["Catch"][0]["Next"] == "MarkFailed"
    assert states["MarkFailed"]["Next"] == "RoundFailed"
    assert states["RoundFailed"]["Type"] == "Fail"


def test_a_round_without_train_skips_the_train_job(full):
    """Change 24: a selection-only round (execution input `train` is null)
    scores with the round's model."""
    states = _definition(full)["States"]
    (choice,) = states["TrainOrNot"]["Choices"]
    assert choice == {"Variable": "$.train", "IsNull": True, "Next": "UseRoundModel"}
    assert states["UseRoundModel"]["Next"] == "Score"
    assert states["UseRoundModel"]["Parameters"] == {"model_uri.$": "$.snapshot.score.model_uri"}
    assert states["UseTrainedModel"]["Parameters"] == {"model_uri.$": "$.train_result.model_uri"}
    for name in ("UseRoundModel", "UseTrainedModel"):
        assert states[name]["ResultPath"] == "$.scoring"
    # Merge gets the whole state: train_result only exists when Train ran.
    # Invoked by function ARN with no Parameters: the event is the whole state.
    assert "Parameters" not in states["Merge"] and states["Merge"]["Resource"] == "ref"


def test_round_waits_for_the_worker_and_the_labels(full):
    states = _definition(full)["States"]
    for job in ("Train", "Score"):
        assert states[job]["Resource"].endswith(":states:::sqs:sendMessage.waitForTaskToken")
        body = states[job]["Parameters"]["MessageBody"]
        assert body["task_token.$"] == "$$.Task.Token"
        assert body["type"] == job.lower()
        # The machine may be off for days: no heartbeat timeout, SQS retention instead.
        assert "HeartbeatSeconds" not in states[job]
        assert states[job]["TimeoutSeconds"] == 14 * 24 * 3600
    assert states["Score"]["Parameters"]["MessageBody"]["spec"]["model_uri.$"] == (
        "$.scoring.model_uri"
    )
    labels = states["AwaitLabels"]
    assert labels["Resource"].endswith(":states:::lambda:invoke.waitForTaskToken")
    assert labels["Parameters"]["Payload"]["task_token.$"] == "$$.Task.Token"


def test_training_state_machine_trains_then_records(full):
    """Change 24: al-train, started by the ops repo's start-training workflow."""
    definition = _definition(full, "al-train")
    states = definition["States"]
    assert _order(definition) == [
        "MarkRunning", "Snapshot", "Train", "Record", "TrainingSucceeded",
    ]  # fmt: skip
    for step in ("MarkRunning", "Snapshot", "Train", "Record"):
        assert states[step]["Catch"][0]["Next"] == "MarkFailed"
    assert states["MarkFailed"]["Next"] == "TrainingFailed"
    train = states["Train"]
    assert train["Resource"].endswith(":states:::sqs:sendMessage.waitForTaskToken")
    body = train["Parameters"]["MessageBody"]
    assert body["type"] == "train" and body["task_token.$"] == "$$.Task.Token"
    assert body["training_id.$"] == "$.training_id" and "round_id" not in body
    assert body["spec.$"] == "$.snapshot.spec"
    assert "HeartbeatSeconds" not in train and train["TimeoutSeconds"] == 14 * 24 * 3600
    assert states["Record"]["Parameters"]["train_result.$"] == "$.train_result"
    assert states["Record"]["Parameters"]["image.$"] == "$.image"
    full.has_resource_properties(
        "AWS::SSM::Parameter", {"Name": "/leaf-al/state-machines/al-train"}
    )
    handlers = {
        p["Properties"]["Handler"]
        for k, p in full.find_resources("AWS::Lambda::Function").items()
        if k.startswith("AlTrain")
    }
    assert handlers == {
        f"lambdas.training.{h}_handler" for h in ("mark_running", "snapshot", "record", "fail")
    }
    full.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "FunctionName": "leaf-al-start-training",
            "Handler": "lambdas.training.start_training_handler",
        },
    )


def test_state_machine_arn_is_in_ssm(full):
    full.has_resource_properties(
        "AWS::SSM::Parameter", {"Name": "/leaf-al/state-machines/al-round"}
    )


# model-release (design doc F5)


def test_release_lambdas_run_in_the_lambda_subnet(full):
    functions = {
        k: v["Properties"]
        for k, v in full.find_resources("AWS::Lambda::Function").items()
        if k.startswith("ModelRelease")
    }
    assert {p["Handler"] for p in functions.values()} == {
        "lambdas.release.prepare_handler",
        "lambdas.release.record_handler",
    }
    for props in functions.values():
        assert props["VpcConfig"]["SubnetIds"] == [{"Ref": "LambdaSubnet"}]
        env = props["Environment"]["Variables"]
        assert env["PG_HOSTADDR"] == {"Fn::GetAtt": ["HostB4E45AD7", "PrivateIp"]}
        assert "{{resolve:secretsmanager:leaf-al/postgres-password" in json.dumps(
            env["PG_PASSWORD"]
        )


def test_release_prepares_exports_then_records(full):
    definition = _definition(full, "model-release")
    states = definition["States"]
    assert _order(definition) == ["Prepare", "Export", "Record", "ReleaseSucceeded"]
    for step in ("Prepare", "Export", "Record"):
        assert states[step]["Catch"][0]["Next"] == "ReleaseFailed"
    assert states["ReleaseFailed"]["Type"] == "Fail"


def test_release_waits_for_the_export_job(full):
    states = _definition(full, "model-release")["States"]
    export = states["Export"]
    assert export["Resource"].endswith(":states:::sqs:sendMessage.waitForTaskToken")
    body = export["Parameters"]["MessageBody"]
    assert body["type"] == "export"
    assert body["task_token.$"] == "$$.Task.Token"
    assert body["job_id.$"] == "$.prepare.job_id"
    assert body["image.$"] == "$.prepare.image"
    assert body["spec.$"] == "$.prepare.spec"
    # The release workflow waits as long as a GitHub job may run.
    assert export["TimeoutSeconds"] == 6 * 3600
    record = states["Record"]["Parameters"]
    assert record["export.$"] == "$.export"
    assert record["commit_sha.$"] == "$.commit_sha"


def test_release_state_machine_arn_is_in_ssm(full):
    full.has_resource_properties(
        "AWS::SSM::Parameter", {"Name": "/leaf-al/state-machines/model-release"}
    )
    assert "ModelReleaseStateMachineArn" in full.to_json()["Outputs"]


def test_budget_only_with_email(full, bare):
    full.has_resource_properties(
        "AWS::Budgets::Budget",
        {"Budget": {"BudgetLimit": {"Amount": 20, "Unit": "USD"}, "TimeUnit": "MONTHLY"}},
    )
    bare.resource_count_is("AWS::Budgets::Budget", 0)


def test_context_overrides():
    t = _template(instanceType="t3.large", dataVolumeGb="30", budgetEmail="a@b.c", budgetUsd="35")
    t.has_resource_properties("AWS::EC2::Instance", {"InstanceType": "t3.large"})
    t.has_resource_properties("AWS::EC2::Volume", {"Size": 30})
    t.has_resource_properties(
        "AWS::Budgets::Budget", {"Budget": {"BudgetLimit": {"Amount": 35, "Unit": "USD"}}}
    )


def test_lambda_libraries_are_pinned_to_the_lock_file():
    from infra.lambda_code import ROUND

    reqs = ROUND.requirements()
    assert {r.split("==")[0] for r in reqs} >= {"psycopg-binary", "pyarrow", "pydantic"}
    assert all("==" in r for r in reqs)


# GitHub Actions (design doc "Control through GitHub")


def _github_role(template: Template) -> dict:
    (role,) = [
        r["Properties"]
        for k, r in template.find_resources("AWS::IAM::Role").items()
        if k.startswith("GithubActionsRole")
    ]
    return role


def test_github_oidc_provider(full):
    (provider,) = full.find_resources("AWS::IAM::OIDCProvider").values()
    assert provider["Properties"]["Url"] == "https://token.actions.githubusercontent.com"
    assert provider["Properties"]["ClientIdList"] == ["sts.amazonaws.com"]


def test_only_main_of_the_ops_repo_may_assume_the_github_role(full):
    role = _github_role(full)
    assert role["RoleName"] == "leaf-al-github-actions"
    (stmt,) = role["AssumeRolePolicyDocument"]["Statement"]
    assert stmt["Action"] == "sts:AssumeRoleWithWebIdentity"
    assert stmt["Condition"] == {
        "StringEquals": {
            "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
            # Not pull requests, other branches or forks: only what merged to
            # main of the ops repo, which holds the round and release files.
            # GitHub's immutable subject names the owner and repo by id too.
            "token.actions.githubusercontent.com:sub": (
                "repo:duytq5@88496947/active-learning-operation@1418488226:ref:refs/heads/main"
            ),
        }
    }
    # The release workflow waits up to 6 hours on the export.
    assert role["MaxSessionDuration"] >= 7 * 3600


def test_github_role_may_only_run_rounds_and_releases(full):
    statements = _statements(full, "GithubActionsRoleDefaultPolicy")
    resources: dict[str, str] = {}
    for s in statements:
        for a in _actions(s):
            resources[a] = resources.get(a, "") + json.dumps(s["Resource"])
    assert set(resources) == {
        "states:StartExecution",
        "states:DescribeExecution",
        "states:StopExecution",
        "ssm:GetParameter",
        "lambda:InvokeFunction",
    }
    start = resources["states:StartExecution"]
    assert "ModelReleaseStateMachine" in start and "AlRoundStateMachine" in start
    assert "AlTrainStateMachine" in start
    describe = resources["states:DescribeExecution"]
    assert ":execution:model-release:*" in describe and ":execution:al-round:*" in describe
    assert ":execution:al-train:*" in describe
    # Cancelling stops a round; releases are never stopped from GitHub.
    assert ":execution:al-round:*" in resources["states:StopExecution"]
    assert "model-release" not in resources["states:StopExecution"]
    params = resources["ssm:GetParameter"]
    assert "/leaf-al/state-machines/model-release" in params
    assert "/leaf-al/state-machines/al-round" in params
    assert "/leaf-al/state-machines/al-train" in params
    invoke = resources["lambda:InvokeFunction"]
    assert "StartRoundFn" in invoke and "CancelRoundFn" in invoke
    assert "StartTrainingFn" in invoke
    assert "GithubActionsRoleArn" in full.to_json()["Outputs"]


def test_the_ops_repo_is_a_setting():
    (stmt,) = _github_role(_template(opsRepo="someone/ops"))["AssumeRolePolicyDocument"][
        "Statement"
    ]
    sub = stmt["Condition"]["StringEquals"]["token.actions.githubusercontent.com:sub"]
    assert sub == "repo:someone/ops:ref:refs/heads/main"


def test_the_ops_repo_subject_is_a_setting():
    template = _template(opsRepoSub="repo:someone@1/ops@2")
    (stmt,) = _github_role(template)["AssumeRolePolicyDocument"]["Statement"]
    sub = stmt["Condition"]["StringEquals"]["token.actions.githubusercontent.com:sub"]
    assert sub == "repo:someone@1/ops@2:ref:refs/heads/main"


def test_round_control_lambdas(full):
    functions = {
        p["FunctionName"]: p
        for p in (f["Properties"] for f in full.find_resources("AWS::Lambda::Function").values())
        if p.get("FunctionName") in ("leaf-al-start-round", "leaf-al-cancel-round")
    }
    assert functions["leaf-al-start-round"]["Handler"] == "lambdas.control.start_round_handler"
    assert functions["leaf-al-cancel-round"]["Handler"] == "lambdas.control.cancel_round_handler"
    for props in functions.values():
        assert props["VpcConfig"]["SubnetIds"] == [{"Ref": "LambdaSubnet"}]
        assert props["Environment"]["Variables"]["PG_HOSTADDR"] == {
            "Fn::GetAtt": ["HostB4E45AD7", "PrivateIp"]
        }
    assert functions["leaf-al-cancel-round"]["Environment"]["Variables"]["DATA_BUCKET"]


def test_an_existing_github_oidc_provider_is_reused():
    arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
    template = _template(githubOidcProviderArn=arn)
    template.resource_count_is("AWS::IAM::OIDCProvider", 0)
    (stmt,) = _github_role(template)["AssumeRolePolicyDocument"]["Statement"]
    assert stmt["Principal"] == {"Federated": arn}
