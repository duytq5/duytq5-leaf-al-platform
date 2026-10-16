"""Image viewer on the EC2 host: FiftyOne behind Caddy and a second
oauth2-proxy (Cognito `admins` only), next to Label Studio. Checks the stack
(domain, Cognito callback), the compose services and the Caddy site."""

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EC2 = ROOT / "deploy" / "ec2"
VIEWER = ("mongo", "viewer-app", "viewer-sync", "viewer-gate")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((EC2 / "docker-compose.yml").read_text())["services"]


def test_viewer_services_publish_no_port(compose):
    for name in VIEWER:
        assert name in compose
        assert "ports" not in compose[name], name  # only Caddy faces the internet


def test_viewer_memory_is_capped(compose):
    """The t3.medium also runs Postgres, MLflow and Label Studio."""
    for name in ("mongo", "viewer-app", "viewer-sync"):
        assert compose[name].get("mem_limit"), name
    assert "--wiredTigerCacheSizeGB" in compose["mongo"]["command"]


def test_viewer_gate_lets_only_admins_through(compose):
    gate = compose["viewer-gate"]
    groups = [a for a in gate["command"] if a.startswith("--allowed-group")]
    assert groups == ["--allowed-group=admins"]
    env = gate["environment"]
    assert env["OAUTH2_PROXY_UPSTREAMS"] == "http://viewer-app:5151"
    assert env["OAUTH2_PROXY_REDIRECT_URL"] == "https://${VIEWER_DOMAIN}/oauth2/callback"
    # Same Cognito client as the Label Studio gate (two callback URLs).
    label_gate = compose["oauth2-proxy"]["environment"]
    for key in ("OAUTH2_PROXY_CLIENT_ID", "OAUTH2_PROXY_OIDC_ISSUER_URL"):
        assert env[key] == label_gate[key]


def test_app_and_sync_share_the_database_and_the_images(compose):
    app, sync = compose["viewer-app"], compose["viewer-sync"]
    for svc in (app, sync):
        assert svc["environment"]["FIFTYONE_DATABASE_URI"] == "mongodb://mongo:27017"
        assert "${DATA_DIR}/viewer/media:/media" in [v.split(":ro")[0] for v in svc["volumes"]]
    assert sync["environment"]["MEDIA_DIR"] == "/media"
    assert "viewer.main" in " ".join(sync["command"])
    assert "--address=0.0.0.0" in app["command"] and "--port=5151" in app["command"]
    # Postgres only over TLS (pg_hba.conf).
    assert "sslmode=require" in sync["environment"]["DATABASE_URL"]


def test_caddy_serves_the_viewer_domain(compose):
    assert compose["caddy"]["environment"]["VIEWER_DOMAIN"].startswith("${VIEWER_DOMAIN")
    caddyfile = (EC2 / "caddy" / "Caddyfile").read_text()
    site = caddyfile.split("{$VIEWER_DOMAIN} {", 1)[1].split("}", 1)[0]
    assert "reverse_proxy viewer-gate:4180" in site


# The stack

pytest.importorskip("aws_cdk")

import aws_cdk as cdk  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402

from infra.stacks.platform import SSM_VIEWER_DOMAIN, PlatformStack  # noqa: E402


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


def test_default_viewer_domain_is_a_subdomain_of_the_elastic_ip_on_sslip(t):
    domain = _param(t, SSM_VIEWER_DOMAIN)
    assert '"view.' in domain and "HostEip" in domain and ".sslip.io" in domain
    other = _template(viewerDomain="view.example.com")
    assert _param(other, SSM_VIEWER_DOMAIN) == '"view.example.com"'


def test_cognito_gate_client_accepts_the_viewer_callback(t):
    (client,) = [
        c["Properties"]
        for c in t.find_resources("AWS::Cognito::UserPoolClient").values()
        if c["Properties"]["ClientName"] == "label-studio"
    ]
    urls = [json.dumps(u) for u in client["CallbackURLs"]]
    assert len(urls) == 3  # Label Studio, the viewer and MLflow
    assert any('"https://view.' in u and "/oauth2/callback" in u for u in urls)
    custom = _template(viewerDomain="view.example.com")
    (client,) = [
        c["Properties"]
        for c in custom.find_resources("AWS::Cognito::UserPoolClient").values()
        if c["Properties"]["ClientName"] == "label-studio"
    ]
    assert "https://view.example.com/oauth2/callback" in client["CallbackURLs"]


def test_viewer_url_output(t):
    assert "ViewerUrl" in t.to_json()["Outputs"]
