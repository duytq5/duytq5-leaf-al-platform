"""MLflow on the internet for ML engineers: Caddy and a third oauth2-proxy
(Cognito `admins` only) in front of it, like the image viewer. The worker keeps
reaching MLflow through the SSM tunnel. Checks the stack (domain, Cognito
callback, API links), the compose services and the Caddy site."""

import fnmatch
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EC2 = ROOT / "deploy" / "ec2"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((EC2 / "docker-compose.yml").read_text())["services"]


def _flag(command: list[str], name: str) -> list[str]:
    (arg,) = [a for a in command if a.startswith(f"--{name}=")]
    return arg.split("=", 1)[1].split(",")


def test_gate_lets_only_admins_through_to_mlflow(compose):
    gate = compose["mlflow-gate"]
    assert "ports" not in gate  # only Caddy faces the internet
    groups = [a for a in gate["command"] if a.startswith("--allowed-group")]
    assert groups == ["--allowed-group=admins"]
    env = gate["environment"]
    assert env["OAUTH2_PROXY_UPSTREAMS"] == "http://mlflow:5000"
    assert env["OAUTH2_PROXY_REDIRECT_URL"] == "https://${MLFLOW_DOMAIN}/oauth2/callback"
    # Same Cognito client as the Label Studio and viewer gates.
    label_gate = compose["oauth2-proxy"]["environment"]
    for key in ("OAUTH2_PROXY_CLIENT_ID", "OAUTH2_PROXY_OIDC_ISSUER_URL"):
        assert env[key] == label_gate[key]


def test_mlflow_accepts_its_domain_and_the_tunnel(compose):
    """MLflow 3 refuses a Host header it does not know (DNS rebinding check)
    and POSTs from another origin; naming hosts replaces its localhost
    defaults, so the tunnel's localhost:<port> must be listed too."""
    command = compose["mlflow"]["command"]
    hosts = _flag(command, "allowed-hosts")
    for host in ("${MLFLOW_DOMAIN}", "localhost:5000", "localhost:5050", "127.0.0.1:5000"):
        assert any(fnmatch.fnmatch(host, h) if "*" in h else host == h for h in hosts), host
    assert "*" not in hosts
    assert _flag(command, "cors-allowed-origins") == ["https://${MLFLOW_DOMAIN}"]
    # The worker and `scripts/ec2.sh tunnel` still use port 5000.
    assert "5000:5000" in compose["mlflow"]["ports"]


def test_caddy_serves_the_mlflow_domain(compose):
    caddy = compose["caddy"]
    assert caddy["environment"]["MLFLOW_DOMAIN"].startswith("${MLFLOW_DOMAIN")
    assert "mlflow-gate" in caddy["depends_on"]
    caddyfile = (EC2 / "caddy" / "Caddyfile").read_text()
    site = caddyfile.split("{$MLFLOW_DOMAIN} {", 1)[1].split("}", 1)[0]
    assert "reverse_proxy mlflow-gate:4180" in site


def test_write_env_reads_the_mlflow_domain():
    script = (EC2 / "write-env.sh").read_text()
    assert "MLFLOW_DOMAIN=$(get mlflow/domain)" in script


# The stack

pytest.importorskip("aws_cdk")

import aws_cdk as cdk  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402

from infra.stacks.platform import SSM_MLFLOW_DOMAIN, PlatformStack  # noqa: E402


def _template(**context) -> Template:
    app = cdk.App(context={"aws:cdk:bundling-stacks": [], **context})
    return Template.from_stack(PlatformStack(app, "LeafAlPlatform"))


@pytest.fixture(scope="module")
def t() -> Template:
    return _template()


def _param(t: Template, name: str) -> str:
    (param,) = [
        p["Properties"]
        for p in t.find_resources("AWS::SSM::Parameter").values()
        if p["Properties"]["Name"] == name
    ]
    return json.dumps(param["Value"])


def _gate_callbacks(t: Template) -> list:
    (client,) = [
        c["Properties"]
        for c in t.find_resources("AWS::Cognito::UserPoolClient").values()
        if c["Properties"]["ClientName"] == "label-studio"
    ]
    return client["CallbackURLs"]


def test_default_mlflow_domain_is_a_subdomain_of_the_elastic_ip_on_sslip(t):
    domain = _param(t, SSM_MLFLOW_DOMAIN)
    assert '"mlflow.' in domain and "HostEip" in domain and ".sslip.io" in domain
    other = _template(mlflowDomain="mlflow.example.com")
    assert _param(other, SSM_MLFLOW_DOMAIN) == '"mlflow.example.com"'


def test_cognito_gate_client_accepts_the_mlflow_callback(t):
    urls = [json.dumps(u) for u in _gate_callbacks(t)]
    assert any('"https://mlflow.' in u and "/oauth2/callback" in u for u in urls)
    custom = _template(mlflowDomain="mlflow.example.com")
    assert "https://mlflow.example.com/oauth2/callback" in _gate_callbacks(custom)


def test_mlflow_url_output_and_api_links(t):
    assert "MlflowUrl" in t.to_json()["Outputs"]
    (api,) = [
        f["Properties"]
        for f in t.find_resources("AWS::Lambda::Function").values()
        if "MLFLOW_URL" in f["Properties"].get("Environment", {}).get("Variables", {})
    ]
    link = json.dumps(api["Environment"]["Variables"]["MLFLOW_URL"])
    assert '"https://mlflow.' in link and "HostEip" in link
