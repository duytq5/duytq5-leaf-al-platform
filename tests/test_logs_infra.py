"""Every log of the stack sits under /leaf-al/ so one search covers the flow."""

import pytest

pytest.importorskip("aws_cdk")

import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.logs import PREFIX
from infra.stacks.platform import PlatformStack

NO_BUNDLING = {"aws:cdk:bundling-stacks": []}


@pytest.fixture(scope="module")
def template() -> Template:
    app = cdk.App(context=NO_BUNDLING)
    return Template.from_stack(PlatformStack(app, "LeafAlPlatform"))


def _groups(template: Template) -> dict[str, dict]:
    return {k: v["Properties"] for k, v in template.find_resources("AWS::Logs::LogGroup").items()}


def _our_functions(template: Template) -> dict[str, dict]:
    # CDK's own helper functions (bucket notifications) are not ours.
    return {
        k: v["Properties"]
        for k, v in template.find_resources("AWS::Lambda::Function").items()
        if not k.startswith("BucketNotificationsHandler")
    }


def test_log_groups_are_named_under_the_prefix(template):
    groups = _groups(template)
    names = [g["LogGroupName"] for g in groups.values()]
    assert PREFIX == "/leaf-al/"
    assert all(n.startswith(PREFIX) for n in names), names
    assert len(names) == len(set(names))
    assert all(g["RetentionInDays"] == 30 for g in groups.values())


def test_named_groups_for_each_part_of_the_flow(template):
    names = {g["LogGroupName"] for g in _groups(template).values()}
    assert {
        "/leaf-al/api",
        "/leaf-al/upload",
        "/leaf-al/post-confirmation",
        "/leaf-al/label-studio-hook",
        "/leaf-al/round/mark-running",
        "/leaf-al/round/snapshot",
        "/leaf-al/round/select",
        "/leaf-al/round/await-labels",
        "/leaf-al/release/prepare",
        "/leaf-al/release/record",
        "/leaf-al/control/start-round",
        "/leaf-al/control/cancel-round",
        "/leaf-al/states/al-round",
        "/leaf-al/states/model-release",
    } <= names


def test_every_lambda_logs_json_to_its_own_group(template):
    groups = _groups(template)
    functions = _our_functions(template)
    assert functions
    used = []
    for name, props in functions.items():
        config = props.get("LoggingConfig", {})
        assert config.get("LogFormat") == "JSON", name
        (ref,) = config["LogGroup"].values()
        assert ref in groups, name
        used.append(ref)
    assert len(used) == len(set(used))


@pytest.mark.parametrize("machine", ["al-round", "model-release"])
def test_state_machines_log_every_event_with_its_data(template, machine):
    groups = _groups(template)
    (props,) = [
        m["Properties"]
        for m in template.find_resources("AWS::StepFunctions::StateMachine").values()
        if m["Properties"]["StateMachineName"] == machine
    ]
    config = props["LoggingConfiguration"]
    assert config["Level"] == "ALL"
    assert config["IncludeExecutionData"] is True
    (dest,) = config["Destinations"]
    (ref, _attr) = dest["CloudWatchLogsLogGroup"]["LogGroupArn"]["Fn::GetAtt"]
    assert groups[ref]["LogGroupName"] == f"/leaf-al/states/{machine}"


def test_saved_queries_cover_every_group(template):
    groups = _groups(template)
    queries = {
        q["Properties"]["Name"]: q["Properties"]
        for q in template.find_resources("AWS::Logs::QueryDefinition").values()
    }
    assert set(queries) == {"leaf-al/follow-an-id", "leaf-al/recent-errors"}
    for q in queries.values():
        assert len(q["LogGroupNames"]) == len(groups)
    assert "ROUND_ID" in queries["leaf-al/follow-an-id"]["QueryString"]
