"""Only Cognito `admins` may create, change or delete Label Studio projects.

Label Studio Community has no roles: every member of the organization can
create projects. So the gate does it. oauth2-proxy sends the person's Cognito
groups in X-Forwarded-Groups (it drops any value the browser sent) to an
internal Caddy listener, which refuses project management to anyone not in
`admins` and passes the rest to Label Studio. The label step Lambdas call
label-studio:8080 directly and never pass this guard.

The behaviour tests run the real Caddy binary (CI copies it from the same
image as EC2); locally they skip when `caddy` is not on PATH."""

import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EC2 = ROOT / "deploy" / "ec2"
GUARD_SITE = "http://:8081 {"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((EC2 / "docker-compose.yml").read_text())["services"]


@pytest.fixture(scope="module")
def caddyfile() -> str:
    return (EC2 / "caddy" / "Caddyfile").read_text()


def _guard_block(caddyfile: str) -> str:
    """The internal site, from its address to its closing brace at column 0."""
    start = caddyfile.index(GUARD_SITE)
    end = caddyfile.index("\n}", start)
    return caddyfile[start : end + 2]


def test_gate_sends_people_through_the_guard(compose):
    env = compose["oauth2-proxy"]["environment"]
    assert env["OAUTH2_PROXY_UPSTREAMS"] == "http://caddy:8081"
    # The guard trusts X-Forwarded-Groups, so oauth2-proxy must set it and
    # must drop whatever the browser sent under that name.
    assert env["OAUTH2_PROXY_PASS_USER_HEADERS"] == "true"
    assert env["OAUTH2_PROXY_SKIP_AUTH_STRIP_HEADERS"] == "true"


def test_guard_listener_is_not_published(compose):
    """Only containers on the compose network reach :8081, never the internet."""
    assert compose["caddy"]["ports"] == ["80:80", "443:443"]


def test_label_step_lambdas_keep_their_direct_port(compose):
    assert compose["label-studio"]["ports"] == ["8080:8080"]


def test_public_site_still_goes_through_the_cognito_gate(caddyfile):
    site = caddyfile.split("{$LS_DOMAIN} {", 1)[1].split("\n}", 1)[0]
    assert "reverse_proxy oauth2-proxy:4180" in site


def test_guard_proxies_to_label_studio(caddyfile):
    block = _guard_block(caddyfile)
    assert "reverse_proxy label-studio:8080" in block
    assert "X-Forwarded-Groups" in block
    assert "admins" in block


# Behaviour, with the real Caddy binary.

CADDY = shutil.which("caddy")
needs_caddy = pytest.mark.skipif(
    CADDY is None and not os.environ.get("CI"), reason="caddy binary not on PATH"
)


def _call(base: str, method: str, path: str, groups: str | None = None) -> tuple[int, str]:
    method = method.upper()
    data = None if method == "GET" else b""
    req = urllib.request.Request(base + path, method=method, data=data)
    if groups is not None:
        req.add_header("X-Forwarded-Groups", groups)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Upstream(BaseHTTPRequestHandler):
    """Stands in for Label Studio: answers 200 to everything."""

    def _ok(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _ok

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def guard(caddyfile, tmp_path_factory):
    if CADDY is None:
        pytest.fail("caddy binary not on PATH (CI must not skip the guard tests)")
    upstream = HTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    port = _free_port()
    block = _guard_block(caddyfile)
    block = block.replace(GUARD_SITE, f"http://127.0.0.1:{port} {{")
    block = block.replace("label-studio:8080", f"127.0.0.1:{upstream.server_port}")
    config = tmp_path_factory.mktemp("caddy") / "Caddyfile"
    config.write_text("{\n\tadmin off\n\tauto_https off\n}\n\n" + block)

    proc = subprocess.Popen(
        [CADDY, "run", "--config", str(config), "--adapter", "caddyfile"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            if proc.poll() is not None:
                pytest.fail(proc.stderr.read().decode())
            time.sleep(0.05)
    yield base
    proc.terminate()
    proc.wait(5)
    upstream.shutdown()


LABELER = "labelers"
ADMIN = "admins"

PROJECT_MANAGEMENT = [
    ("POST", "/api/projects"),
    ("POST", "/api/projects/"),
    ("PATCH", "/api/projects/3"),
    ("PUT", "/api/projects/3/"),
    ("DELETE", "/api/projects/3/"),
    ("POST", "/api/projects/3/import"),
    ("POST", "/api/projects/3/reimport"),
    ("POST", "/api/projects/3/duplicate"),
    ("DELETE", "/api/projects/3/tasks/"),
    ("POST", "/api/import/file-upload"),
    ("POST", "/api/storages/s3"),
    ("POST", "/api/storages/s3/1/sync"),
    ("DELETE", "/api/storages/s3/1"),
    ("POST", "/api/webhooks/"),
    ("PATCH", "/api/webhooks/1"),
    ("post", "/API/Projects"),
]

LABELING = [
    ("GET", "/api/projects"),
    ("GET", "/api/projects/3/"),
    ("GET", "/api/projects/3/next"),
    ("GET", "/api/projects/3/label-stream-history"),
    ("POST", "/api/tasks/7/annotations"),
    ("PATCH", "/api/annotations/9"),
    ("DELETE", "/api/annotations/9"),
    ("POST", "/api/tasks/7/drafts"),
    ("PATCH", "/api/drafts/4"),
    ("POST", "/api/dm/actions?id=next_task&project=3"),
    ("POST", "/api/dm/views"),
    ("PATCH", "/api/current-user"),
    ("GET", "/projects/"),
    ("POST", "/user/logout"),
]


@needs_caddy
@pytest.mark.parametrize(("method", "path"), PROJECT_MANAGEMENT)
def test_labelers_cannot_manage_projects(guard, method, path):
    status, _ = _call(guard, method, path, LABELER)
    assert status == 403, (method, path)


@needs_caddy
@pytest.mark.parametrize(("method", "path"), PROJECT_MANAGEMENT)
def test_admins_can_manage_projects(guard, method, path):
    status, _ = _call(guard, method, path, ADMIN)
    assert status == 200, (method, path)


@needs_caddy
@pytest.mark.parametrize(("method", "path"), LABELING)
def test_labelers_can_label(guard, method, path):
    status, _ = _call(guard, method, path, LABELER)
    assert status == 200, (method, path)


@needs_caddy
@pytest.mark.parametrize(
    "groups",
    ["", "notadmins", "admins2", "labelers,superadmins", "admin"],
)
def test_only_the_exact_admins_group_counts(guard, groups):
    assert _call(guard, "POST", "/api/projects", groups)[0] == 403, groups


@needs_caddy
@pytest.mark.parametrize("groups", ["labelers,admins", "admins,labelers", "labelers, admins"])
def test_admins_in_a_group_list_counts(guard, groups):
    assert _call(guard, "POST", "/api/projects", groups)[0] == 200, groups


@needs_caddy
def test_no_groups_header_cannot_manage_projects(guard):
    assert _call(guard, "POST", "/api/projects")[0] == 403


@needs_caddy
def test_refusal_says_why(guard):
    status, body = _call(guard, "POST", "/api/projects", LABELER)
    assert status == 403 and re.search(r"admin", body, re.I)
