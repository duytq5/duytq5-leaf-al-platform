"""Label Studio infrastructure (F2): the public HTTPS entry on the EC2 host, its
Cognito client, the label step and the webhook Lambdas."""

import json

import pytest

pytest.importorskip("aws_cdk")

import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.stacks.platform import (
    LABEL_STUDIO_SECRET,
    SSM_LABEL_STUDIO_CLIENT_ID,
    SSM_LABEL_STUDIO_DOMAIN,
    SSM_USER_POOL_ID,
    PlatformStack,
)

NO_BUNDLING = {"aws:cdk:bundling-stacks": []}


def _template(**context) -> Template:
    app = cdk.App(context={**NO_BUNDLING, **context})
    return Template.from_stack(PlatformStack(app, "LeafAlPlatform"))


@pytest.fixture(scope="module")
def t() -> Template:
    return _template()


def _functions(t: Template) -> dict[str, dict]:
    return {
        f["Properties"]["Handler"]: f["Properties"]
        for f in t.find_resources("AWS::Lambda::Function").values()
        if "Handler" in f["Properties"]
    }


def _ingress(t: Template) -> list[dict]:
    """Every inbound rule of the host: inline (CIDR) and separate (group) ones."""
    (host,) = [
        g for k, g in t.find_resources("AWS::EC2::SecurityGroup").items() if k.startswith("HostSg")
    ]
    inline = host["Properties"].get("SecurityGroupIngress", [])
    rules = [r["Properties"] for r in t.find_resources("AWS::EC2::SecurityGroupIngress").values()]
    return inline + rules


def test_host_takes_https_from_anyone_and_label_studio_only_from_the_lambdas(t):
    rules = {(r["FromPort"], r.get("CidrIp")): r for r in _ingress(t)}
    assert set(rules) == {(80, "0.0.0.0/0"), (443, "0.0.0.0/0"), (5432, None), (8080, None)}
    ls = rules[(8080, None)]
    assert ls["SourceSecurityGroupId"]["Fn::GetAtt"][0].startswith("LambdaSg")
    # Postgres, MLflow and SSH stay closed to the internet.
    assert not {5000, 5432, 22} & {r["FromPort"] for r in _ingress(t) if "CidrIp" in r}


def test_host_keeps_its_address_across_stop_and_start(t):
    (eip,) = t.find_resources("AWS::EC2::EIP").values()
    assert eip["Properties"]["Domain"] == "vpc"
    (assoc,) = t.find_resources("AWS::EC2::EIPAssociation").values()
    assert assoc["Properties"]["InstanceId"]["Ref"].startswith("Host")


def _domain(t: Template) -> str:
    (param,) = [
        p["Properties"]
        for p in t.find_resources("AWS::SSM::Parameter").values()
        if p["Properties"]["Name"] == SSM_LABEL_STUDIO_DOMAIN
    ]
    return json.dumps(param["Value"])


def test_default_domain_is_the_elastic_ip_on_sslip(t):
    domain = _domain(t)
    assert "HostEip" in domain and ".sslip.io" in domain
    assert _template(labelStudioDomain="label.example.com").find_resources(
        "AWS::SSM::Parameter", {"Properties": {"Value": "label.example.com"}}
    )


def test_label_studio_cognito_client_is_confidential_and_code_flow_only(t):
    clients = {
        c["Properties"]["ClientName"]: c["Properties"]
        for c in t.find_resources("AWS::Cognito::UserPoolClient").values()
    }
    client = clients["label-studio"]
    assert client["GenerateSecret"] is True  # oauth2-proxy runs on a server
    assert client["AllowedOAuthFlows"] == ["code"]
    assert client["SupportedIdentityProviders"] == ["COGNITO"]  # no Google here
    assert client["ExplicitAuthFlows"] == ["ALLOW_REFRESH_TOKEN_AUTH"]
    callback = json.dumps(client["CallbackURLs"])
    assert "https://" in callback and "/oauth2/callback" in callback
    names = {p["Properties"]["Name"] for p in t.find_resources("AWS::SSM::Parameter").values()}
    assert {SSM_USER_POOL_ID, SSM_LABEL_STUDIO_CLIENT_ID, SSM_LABEL_STUDIO_DOMAIN} <= names


def test_host_reads_the_label_studio_settings_and_images(t):
    statements = [
        s
        for k, p in t.find_resources("AWS::IAM::Policy").items()
        if k.startswith("HostLabelStudio")
        for s in p["Properties"]["PolicyDocument"]["Statement"]
    ]
    by_action = {json.dumps(s["Action"]): s for s in statements}
    secret = by_action['"secretsmanager:GetSecretValue"']
    assert f":secret:{LABEL_STUDIO_SECRET}-" in json.dumps(secret["Resource"])
    client = by_action['"cognito-idp:DescribeUserPoolClient"']
    assert "UserPool" in json.dumps(client["Resource"])
    # Label Studio signs image links with the instance role: raw/ only.
    read = by_action['"s3:GetObject"']
    assert json.dumps(read["Resource"]).endswith('"/raw/*"]]}')
    listing = by_action['"s3:ListBucket"']
    assert listing["Condition"] == {"StringLike": {"s3:prefix": ["raw/*"]}}


def test_label_step_and_record_lambda_run_in_the_vpc(t):
    fns = _functions(t)
    start = fns["lambdas.labeling.start_handler"]
    record = fns["lambdas.labeling.record_handler"]
    for fn in (start, record):
        assert fn["VpcConfig"]["SubnetIds"] == [{"Ref": "LambdaSubnet"}]
        env = fn["Environment"]["Variables"]
        assert ":8080" in json.dumps(env["LS_URL"])
        assert "PrivateIp" in json.dumps(env["LS_URL"])
        assert f"{{{{resolve:secretsmanager:{LABEL_STUDIO_SECRET}:SecretString:api_token" in (
            json.dumps(env["LS_API_TOKEN"])
        )
    env = start["Environment"]["Variables"]
    assert "/v1/hooks/label-studio" in json.dumps(env["LS_WEBHOOK_URL"])
    assert ":SecretString:webhook_secret" in json.dumps(env["LS_WEBHOOK_SECRET"])


def test_webhook_lambda_is_outside_the_vpc_and_can_resume_rounds(t):
    hook = _functions(t)["lambdas.label_studio_hook.handler"]
    assert "VpcConfig" not in hook  # it needs the Step Functions API (no NAT)
    env = hook["Environment"]["Variables"]
    assert env["RECORD_LABEL_FUNCTION"]["Ref"].startswith("AlRoundRecordLabelFn")
    assert ":SecretString:webhook_secret" in json.dumps(env["LS_WEBHOOK_SECRET"])
    statements = [
        s
        for k, p in t.find_resources("AWS::IAM::Policy").items()
        if k.startswith("LabelStudioHookFn")
        for s in p["Properties"]["PolicyDocument"]["Statement"]
    ]
    actions = {a for s in statements for a in ([s["Action"]] if isinstance(s["Action"], str)
                                               else s["Action"])}  # fmt: skip
    assert "states:SendTaskSuccess" in actions
    assert "lambda:InvokeFunction" in actions
    assert "states:SendTaskFailure" not in actions


def test_webhook_route_has_no_jwt_but_everything_else_does(t):
    routes = {
        r["Properties"]["RouteKey"]: r["Properties"]
        for r in t.find_resources("AWS::ApiGatewayV2::Route").values()
    }
    assert routes["POST /v1/hooks/label-studio"]["AuthorizationType"] == "NONE"
    assert routes["ANY /v1/{proxy+}"]["AuthorizationType"] == "JWT"
    # The full route list, with the app's public routes: tests/test_api_infra.py.


def test_label_step_is_retried_while_label_studio_restarts(t):
    (machine,) = t.find_resources("AWS::StepFunctions::StateMachine").values()
    parts = machine["Properties"]["DefinitionString"]["Fn::Join"][1]
    states = json.loads("".join(p if isinstance(p, str) else "ref" for p in parts))["States"]
    errors = [e for r in states["AwaitLabels"]["Retry"] for e in r["ErrorEquals"]]
    assert "LabelStudioError" in errors


def test_label_studio_url_output(t):
    assert "LabelStudioUrl" in t.to_json()["Outputs"]
