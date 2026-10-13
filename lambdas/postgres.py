"""Postgres connection for Lambdas inside the VPC.

The Lambdas reach the EC2 host on its private IP (PG_HOSTADDR). The server
certificate names leaf-al-postgres, not the IP, so libpq checks the name
(host) while connecting to the address (hostaddr), against the pinned CA.
The Lambdas have no internet route, so CloudFormation puts the password
(PG_PASSWORD, from Secrets Manager) and the CA (PG_CA, from SSM) into their
settings at deploy time.

Tests and local runs set DATABASE_URL instead.
"""

import os
from functools import cache
from pathlib import Path

import psycopg

from db import connect

PG_HOSTNAME = "leaf-al-postgres"
CA_PATH = Path("/tmp/postgres-ca.crt")


@cache
def _conninfo() -> str:
    CA_PATH.write_text(os.environ["PG_CA"])
    return psycopg.conninfo.make_conninfo(
        host=PG_HOSTNAME,
        hostaddr=os.environ["PG_HOSTADDR"],
        port=5432,
        dbname="al",
        user="al",
        password=os.environ["PG_PASSWORD"],
        sslmode="verify-full",
        sslrootcert=str(CA_PATH),
        connect_timeout=10,
    )


def open_db() -> psycopg.Connection:
    """An autocommit connection; group writes with `conn.transaction()`."""
    url = os.environ.get("DATABASE_URL")
    return connect(url) if url else connect(_conninfo())
