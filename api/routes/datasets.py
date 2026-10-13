"""Dataset versions for admins: what each round added, and what a version holds.

A version is the set of (image, label) rows in dataset_version_labels; v0 is
the seed, each succeeded round makes the next one.
"""

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Path, Query

from api.auth import User, require_groups
from api.deps import get_db
from common.admin import (
    DatasetVersion,
    DatasetVersionsResponse,
    VersionItem,
    VersionItemsResponse,
)
from common.types import DatasetName

router = APIRouter()

MAX_PAGE = 1000

_VERSIONS = """
    SELECT v.id, v.version, p.version, v.created_by_round_id, v.created_at
      FROM dataset_versions v
      LEFT JOIN dataset_versions p ON p.id = v.parent_version_id
     WHERE v.dataset_id = %s
     ORDER BY v.version DESC
"""

# Every class of the dataset for every version, 0 included.
_COUNTS = """
    WITH n AS (
        SELECT dvl.version_id, l.class_id, count(*) AS n
          FROM dataset_versions v
          JOIN dataset_version_labels dvl ON dvl.version_id = v.id
          JOIN labels l ON l.id = dvl.label_id
         WHERE v.dataset_id = %(dataset)s
         GROUP BY dvl.version_id, l.class_id
    )
    SELECT v.id, c.code, coalesce(n.n, 0)
      FROM dataset_versions v
      JOIN classes c ON c.dataset_id = v.dataset_id
      LEFT JOIN n ON n.version_id = v.id AND n.class_id = c.id
     WHERE v.dataset_id = %(dataset)s
     ORDER BY c.position
"""

_ITEMS = """
    SELECT i.id, i.s3_key, i.split, c.code, l.source, l.round_id
      FROM dataset_version_labels dvl
      JOIN images i ON i.id = dvl.image_id
      JOIN labels l ON l.id = dvl.label_id
      JOIN classes c ON c.id = l.class_id
     WHERE dvl.version_id = %s AND dvl.image_id > %s
     ORDER BY dvl.image_id
     LIMIT %s
"""


def _dataset_id(db: psycopg.Connection, name: str) -> int:
    row = db.execute("SELECT id FROM datasets WHERE name = %s", (name,)).fetchone()
    if row is None:
        raise HTTPException(404, f"No dataset named {name}.")
    return row[0]


@router.get("/v1/datasets/{name}/versions")
def dataset_versions(
    name: DatasetName,
    _: Annotated[User, Depends(require_groups("admins"))],
    db: Annotated[psycopg.Connection, Depends(get_db)],
) -> DatasetVersionsResponse:
    """Every version of the dataset, newest first, with its labels per class."""
    dataset_id = _dataset_id(db, name)
    counts: dict[int, dict[str, int]] = {}
    for version_id, code, n in db.execute(_COUNTS, {"dataset": dataset_id}):
        counts.setdefault(version_id, {})[code] = n
    versions = [
        DatasetVersion(
            version=version,
            parent_version=parent,
            created_by_round=round_id,
            created_at=created_at,
            total=sum(counts.get(version_id, {}).values()),
            label_counts=counts.get(version_id, {}),
        )
        for version_id, version, parent, round_id, created_at in db.execute(
            _VERSIONS, (dataset_id,)
        )
    ]
    return DatasetVersionsResponse(dataset=name, versions=versions)


@router.get("/v1/datasets/{name}/versions/{version}")
def version_items(
    name: DatasetName,
    version: Annotated[int, Path(ge=0)],
    _: Annotated[User, Depends(require_groups("admins"))],
    db: Annotated[psycopg.Connection, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 100,
    cursor: Annotated[int, Query(ge=0)] = 0,
) -> VersionItemsResponse:
    """The images of one version with their labels, by image id, a page at a time."""
    dataset_id = _dataset_id(db, name)
    row = db.execute(
        "SELECT id FROM dataset_versions WHERE dataset_id = %s AND version = %s",
        (dataset_id, version),
    ).fetchone()
    if row is None:
        raise HTTPException(404, f"{name} has no version {version}.")
    rows = db.execute(_ITEMS, (row[0], cursor, limit + 1)).fetchall()
    page = rows[:limit]
    items = [
        VersionItem(
            image_id=image_id,
            s3_key=s3_key,
            split=split,
            class_code=code,
            label_source=source,
            round_id=round_id,
        )
        for image_id, s3_key, split, code, source, round_id in page
    ]
    more = len(rows) > limit
    return VersionItemsResponse(
        dataset=name,
        version=version,
        items=items,
        next_cursor=str(page[-1][0]) if more else None,
    )
