import json
import shutil
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("aws_cdk")

import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.lambda_code import API, ROUND
from infra.stacks.platform import PlatformStack

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
    routes = {
        r["Properties"]["RouteKey"]: r["Properties"]["AuthorizationType"]
        for r in t.find_resources("AWS::ApiGatewayV2::Route").values()
    }
    # The Label Studio webhook proves itself with a shared secret instead
    # (tests/test_label_studio_infra.py).
    assert routes == {"ANY /v1/{proxy+}": "JWT", "POST /v1/hooks/label-studio": "NONE"}


def test_stage_is_throttled(t):
    (stage,) = t.find_resources("AWS::ApiGatewayV2::Stage").values()
    settings = stage["Properties"]["DefaultRouteSettings"]
    assert settings["ThrottlingRateLimit"] == 10
    assert settings["ThrottlingBurstLimit"] == 20
    assert stage["Properties"]["AutoDeploy"] is True


def test_api_output(t):
    assert "ApiUrl" in t.to_json()["Outputs"]


def test_bundles_are_pinned_to_the_lock_file():
    lock = {
        p["name"]: p["version"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    }
    for bundle in (API, ROUND):
        for name, version in (r.split("==") for r in bundle.requirements()):
            assert lock[name] == version
    # The API bundle stays small: no Parquet or numpy.
    assert {r.split("==")[0] for r in API.requirements()} >= {"fastapi", "mangum"}
    assert not {"numpy", "pyarrow"} & set(API.libraries)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_local_bundle_has_linux_wheels_and_code(tmp_path):
    # Downloads wheels (needs network): the same step `cdk deploy` runs.
    API.build(tmp_path, shutil.which("uv"))
    assert (tmp_path / "api" / "app.py").is_file()
    assert (tmp_path / "fastapi").is_dir() and (tmp_path / "mangum").is_dir()
    assert not list((tmp_path / "api").rglob("__pycache__"))
    assert not (tmp_path / "numpy").exists()
    (core,) = list(tmp_path.glob("pydantic_core/_pydantic_core*.so"))
    assert "x86_64-linux-gnu" in core.name and "cpython-312" in core.name
