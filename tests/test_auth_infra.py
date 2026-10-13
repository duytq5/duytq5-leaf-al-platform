import json

import pytest

pytest.importorskip("aws_cdk")

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from infra.stacks.platform import AUTH_GROUPS, PlatformStack


def _template(**context) -> Template:
    app = cdk.App(context=context)
    return Template.from_stack(PlatformStack(app, "LeafAlPlatform"))


@pytest.fixture(scope="module")
def bare() -> Template:
    return _template()


@pytest.fixture(scope="module")
def google() -> Template:
    return _template(googleClientId="123.apps.googleusercontent.com")


def _clients(template: Template) -> dict[str, dict]:
    clients = template.find_resources("AWS::Cognito::UserPoolClient")
    return {c["Properties"]["ClientName"]: c["Properties"] for c in clients.values()}


def test_user_pool_owner_creates_accounts(bare):
    (pool,) = bare.find_resources("AWS::Cognito::UserPool").values()
    props = pool["Properties"]
    assert props["AdminCreateUserConfig"]["AllowAdminCreateUserOnly"] is True
    assert props["UsernameAttributes"] == ["email"]
    assert props["Policies"]["PasswordPolicy"]["MinimumLength"] == 12
    # TOTP is optional for the pool; the owner turns it on for admin accounts.
    assert props["MfaConfiguration"] == "OPTIONAL"
    assert props["EnabledMfas"] == ["SOFTWARE_TOKEN_MFA"]
    # Losing the pool would lose every account.
    assert pool["DeletionPolicy"] == "Retain"


def test_three_groups():
    assert AUTH_GROUPS == ("admins", "labelers", "uploaders")


def test_groups_created(bare):
    groups = bare.find_resources("AWS::Cognito::UserPoolGroup")
    assert sorted(g["Properties"]["GroupName"] for g in groups.values()) == sorted(AUTH_GROUPS)


def test_public_pkce_clients(bare):
    clients = _clients(bare)
    # label-studio is oauth2-proxy's server-side client (tests/test_label_studio_infra.py).
    assert set(clients) == {"admin-web", "mobile", "label-studio"}
    for name, props in clients.items():
        if name == "label-studio":
            continue
        assert props.get("GenerateSecret", False) is False  # public client, PKCE
        assert props["AllowedOAuthFlows"] == ["code"]
        assert props["AllowedOAuthFlowsUserPoolClient"] is True
        assert set(props["AllowedOAuthScopes"]) == {"openid", "email", "profile"}
        # Sign-in only through the hosted login page, no password API.
        assert props["ExplicitAuthFlows"] == ["ALLOW_REFRESH_TOKEN_AUTH"]
        assert props["PreventUserExistenceErrors"] == "ENABLED"
        assert props["AccessTokenValidity"] == 60
        assert props["RefreshTokenValidity"] == 43200  # 30 days in minutes
        assert props["TokenValidityUnits"]["AccessToken"] == "minutes"


def test_callback_defaults_and_overrides(bare):
    clients = _clients(bare)
    assert clients["mobile"]["CallbackURLs"] == ["leafal://auth/callback"]
    assert clients["admin-web"]["CallbackURLs"] == ["http://localhost:8080/auth/callback"]
    t = _template(
        mobileCallbackUrls="leafal://cb,exp://127.0.0.1:8081/--/cb",
        adminCallbackUrls="https://admin.example.com/cb",
    )
    clients = _clients(t)
    assert clients["mobile"]["CallbackURLs"] == ["leafal://cb", "exp://127.0.0.1:8081/--/cb"]
    assert clients["admin-web"]["CallbackURLs"] == ["https://admin.example.com/cb"]


def test_no_google_without_client_id(bare):
    bare.resource_count_is("AWS::Cognito::UserPoolIdentityProvider", 0)
    for props in _clients(bare).values():
        assert props["SupportedIdentityProviders"] == ["COGNITO"]


def test_google_sign_in_for_mobile(google):
    (idp,) = google.find_resources("AWS::Cognito::UserPoolIdentityProvider").values()
    props = idp["Properties"]
    assert props["ProviderName"] == "Google"
    details = props["ProviderDetails"]
    assert details["client_id"] == "123.apps.googleusercontent.com"
    # The secret stays out of the template: resolved by CloudFormation at deploy.
    assert "resolve:secretsmanager:leaf-al/google-client-secret" in json.dumps(
        details["client_secret"]
    )
    assert "openid" in details["authorize_scopes"]
    assert props["AttributeMapping"]["email"] == "email"
    clients = _clients(google)
    assert clients["mobile"]["SupportedIdentityProviders"] == ["COGNITO", "Google"]
    assert clients["admin-web"]["SupportedIdentityProviders"] == ["COGNITO"]
    # The client must wait for the provider, or CloudFormation rejects it.
    mobile_id = next(
        k
        for k, v in google.find_resources("AWS::Cognito::UserPoolClient").items()
        if v["Properties"]["ClientName"] == "mobile"
    )
    deps = google.to_json()["Resources"][mobile_id].get("DependsOn", [])
    assert any(d.startswith("UserPoolGoogle") for d in deps)


def test_hosted_login_domain(bare):
    (domain,) = bare.find_resources("AWS::Cognito::UserPoolDomain").values()
    assert "leaf-al-" in json.dumps(domain["Properties"]["Domain"])
    t = _template(authDomainPrefix="my-leaf-al")
    t.has_resource_properties("AWS::Cognito::UserPoolDomain", {"Domain": "my-leaf-al"})


def test_post_confirmation_adds_google_users_to_uploaders(bare):
    (pool,) = bare.find_resources("AWS::Cognito::UserPool").values()
    trigger = pool["Properties"]["LambdaConfig"]["PostConfirmation"]
    fn_id = trigger["Fn::GetAtt"][0]
    fn = bare.to_json()["Resources"][fn_id]["Properties"]
    assert fn["Handler"] == "handler.handler"
    assert fn["Runtime"].startswith("python3.")
    assert "VpcConfig" not in fn
    # The function may only add users to groups, nothing else.
    policies = bare.find_resources(
        "AWS::IAM::Policy",
        {"Properties": {"Roles": [{"Ref": Match.string_like_regexp("PostConfirmation")}]}},
    )
    (policy,) = policies.values()
    (stmt,) = policy["Properties"]["PolicyDocument"]["Statement"]
    assert stmt["Action"] == "cognito-idp:AdminAddUserToGroup"
    bare.has_resource_properties(
        "AWS::Lambda::Permission",
        {"Action": "lambda:InvokeFunction", "Principal": "cognito-idp.amazonaws.com"},
    )
    bare.has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 30})


def test_auth_outputs(bare):
    outputs = bare.to_json()["Outputs"]
    for key in ("UserPoolId", "MobileClientId", "AdminWebClientId", "AuthDomainUrl"):
        assert key in outputs
