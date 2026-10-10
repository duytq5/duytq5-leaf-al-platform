"""Postgres access for the CLI and Lambdas.

The connection string comes from DATABASE_URL. Postgres on EC2 only accepts
TLS connections from outside its host, so a remote URL looks like:

  postgresql://al:<password>@<ec2-ip>:5432/al?sslmode=verify-ca&sslrootcert=postgres-server.crt

The server certificate is self-signed and pinned with sslrootcert (it is in
SSM at /leaf-al/postgres/tls-cert); see docs/aws-setup.md.
"""

import os

import psycopg


def connect(url: str | None = None) -> psycopg.Connection:
    """Open an autocommit connection; group writes with `conn.transaction()`."""
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("set DATABASE_URL (see docs/aws-setup.md)")
    return psycopg.connect(url, autocommit=True)
