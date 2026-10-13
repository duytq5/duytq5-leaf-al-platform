import json
import shutil
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("aws_cdk")

import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.stacks.platform import API_DIR, PlatformStack, UvBundler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def t() -> Template:
    return Template.from_stack(PlatformStack(cdk.App(), "LeafAlPlatform"))


def _api_fn(t: Template) -> dict:
    (fn,) = [
        f["Properties"]
        for f in t.find_resources("AWS::Lambda::Function").values()
        if f["Properties"].get("Handler") == "api.app.handler"
    ]
    return fn


def test_api_lambda(t):
    fn = _api_fn(t)
    assert fn["Runtime"] == "python3.12"
    assert fn["Architectures"] == ["x86_64"]  # matches the bundled wheels
    assert "VpcConfig" not in fn
    # Reached only through API Gateway: no public function URL.
    t.resource_count_is("AWS::Lambda::Url", 0)
    t.has_resource_properties(
        "AWS::Lambda::Permission",
        {"Action": "lambda:InvokeFunction", "Principal": "apigateway.amazonaws.com"},
    )


def test_http_api_with_cognito_jwt_authorizer(t):
    t.has_resource_properties("AWS::ApiGatewayV2::Api", {"ProtocolType": "HTTP"})
    (auth,) = t.find_resources("AWS::ApiGatewayV2::Authorizer").values()
    props = auth["Properties"]
    assert props["AuthorizerType"] == "JWT"
    assert props["IdentitySource"] == ["$request.header.Authorization"]
    issuer = json.dumps(props["JwtConfiguration"]["Issuer"])
    assert "cognito-idp." in issuer and '"Ref": "UserPool' in issuer
    audience = json.dumps(props["JwtConfiguration"]["Audience"])
    assert '"Ref": "UserPoolMobileClient' in audience
    assert '"Ref": "UserPoolAdminWebClient' in audience


def test_every_route_needs_a_token(t):
    routes = t.find_resources("AWS::ApiGatewayV2::Route")
    assert routes
    for route in routes.values():
        assert route["Properties"]["AuthorizationType"] == "JWT"
    keys = sorted(r["Properties"]["RouteKey"] for r in routes.values())
    assert keys == ["ANY /v1/{proxy+}"]


def test_stage_is_throttled(t):
    (stage,) = t.find_resources("AWS::ApiGatewayV2::Stage").values()
    settings = stage["Properties"]["DefaultRouteSettings"]
    assert settings["ThrottlingRateLimit"] == 10
    assert settings["ThrottlingBurstLimit"] == 20
    assert stage["Properties"]["AutoDeploy"] is True


def test_api_output(t):
    assert "ApiUrl" in t.to_json()["Outputs"]


def test_bundle_pins_match_lock():
    lock = {
        p["name"]: p["version"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    }
    pins = [
        line.split("==")
        for line in (API_DIR / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert {name for name, _ in pins} == {"fastapi", "mangum"}
    for name, version in pins:
        assert lock[name] == version, f"{name}: requirements.txt {version}, uv.lock {lock[name]}"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_local_bundle_has_linux_wheels_and_code(tmp_path):
    # Downloads wheels (needs network): the same step `cdk deploy` runs.
    assert UvBundler(API_DIR).try_bundle(str(tmp_path), None) is True
    assert (tmp_path / "api" / "app.py").is_file()
    assert (tmp_path / "fastapi").is_dir() and (tmp_path / "mangum").is_dir()
    assert not list((tmp_path / "api").rglob("__pycache__"))
    (core,) = list(tmp_path.glob("pydantic_core/_pydantic_core*.so"))
    assert "x86_64-linux-gnu" in core.name and "cpython-312" in core.name
