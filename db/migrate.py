"""Apply the SQL files in db/migrations/ in name order, each exactly once.

Safe to run again and from two places at once: an advisory lock serializes
runners, and applied files are recorded in schema_migrations. Each file runs
in its own transaction together with its record.
"""

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_ID = 0x1EAF_A1  # any constant shared by all runners


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations; return the names of the ones applied now.

    `conn` must be in autocommit mode (db.connect() returns one).
    """
    applied_now: list[str] = []
    with conn.transaction():
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
    conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_ID,))
    try:
        done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
        for path in migration_files():
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied_now.append(path.name)
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_ID,))
    return applied_now
