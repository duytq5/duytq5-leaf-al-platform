"""deploy/ec2/make-tls.sh: named-curve EC keys that OpenSSL 3 (libpq) accepts.

macOS's openssl (LibreSSL) writes EC keys with explicit curve parameters unless
told otherwise, and OpenSSL 3 then rejects the certificates under
sslmode=verify-full ("Certificate public key has explicit ECC parameters").
The stub openssl below wraps the real one and behaves like LibreSSL: a
`req -newkey` without `ec_param_enc:named_curve` gets explicit parameters.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "ec2" / "make-tls.sh"
REAL_OPENSSL = shutil.which("openssl")

LIBRESSL_LIKE = f"""#!/bin/sh
if [ "$1" = req ] && ! printf '%s\\n' "$@" | grep -qx ec_param_enc:named_curve; then
  exec {REAL_OPENSSL} "$@" -pkeyopt ec_param_enc:explicit
fi
exec {REAL_OPENSSL} "$@"
"""

# Ignores any request for named curves: the script must notice and fail.
ALWAYS_EXPLICIT = f"""#!/bin/sh
if [ "$1" = req ]; then
  exec {REAL_OPENSSL} "$@" -pkeyopt ec_param_enc:explicit
fi
exec {REAL_OPENSSL} "$@"
"""


def run(tmp_path, stub):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    f = bin_dir / "openssl"
    f.write_text(stub)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "tls"
    proc = subprocess.run(
        ["sh", str(SCRIPT), str(out)],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, out


def openssl(*args):
    return subprocess.run([REAL_OPENSSL, *args], capture_output=True, text=True, check=True).stdout


def test_certificates_use_the_named_p256_curve_and_verify(tmp_path):
    proc, out = run(tmp_path, LIBRESSL_LIKE)
    assert proc.returncode == 0, proc.stderr
    for cert in ("ca.crt", "server.crt"):
        assert "ASN1 OID: prime256v1" in openssl("x509", "-in", str(out / cert), "-noout", "-text")
    assert "ASN1 OID: prime256v1" in openssl(
        "pkey", "-in", str(out / "server.key"), "-noout", "-text"
    )
    verify = subprocess.run(
        [REAL_OPENSSL, "verify", "-CAfile", str(out / "ca.crt"), str(out / "server.crt")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_fails_loudly_when_a_key_has_explicit_parameters(tmp_path):
    proc, _ = run(tmp_path, ALWAYS_EXPLICIT)
    assert proc.returncode != 0
    assert "explicit" in proc.stderr
