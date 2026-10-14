"""One row per image of a dataset, with what an operator wants to filter on:
split, status, source, its label in the latest version, the versions that hold
it, a label from a round that has not merged yet, and the app's guess."""

from datetime import datetime
from uuid import UUID

import psycopg
from pydantic import BaseModel, ConfigDict

from common.types import ClassCode, ImageSource, ImageStatus, Split


class ImageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: int
    s3_key: str
    split: Split
    status: ImageStatus
    source: ImageSource
    created_at: datetime
    # Its label in the latest version (what the next round trains on).
    label: ClassCode | None
    label_source: str | None
    labeled_by: str | None
    label_round: UUID | None
    versions: list[int]  # every version that holds the image, ascending
    # The newest label when no version holds it yet: labeled in a running round.
    new_label: ClassCode | None
    new_label_round: UUID | None
    # The on-device model's top class for an app upload, from its capture.
    app_prediction: ClassCode | None
    app_confidence: float | None


_RECORDS = """
    WITH latest AS (
        SELECT id FROM dataset_versions
         WHERE dataset_id = %(dataset)s ORDER BY version DESC LIMIT 1
    )
    SELECT i.id, i.s3_key, i.split, i.status, i.source, i.created_at,
           c.code, l.source, l.labeled_by, l.round_id,
           coalesce(v.versions, '{}'),
           nc.code, n.round_id,
           pc.code, p.confidence
      FROM images i
      LEFT JOIN dataset_version_labels dvl
             ON dvl.image_id = i.id AND dvl.version_id = (SELECT id FROM latest)
      LEFT JOIN labels l ON l.id = dvl.label_id
      LEFT JOIN classes c ON c.id = l.class_id
      LEFT JOIN LATERAL (
          SELECT array_agg(dv.version ORDER BY dv.version) AS versions
            FROM dataset_version_labels x
            JOIN dataset_versions dv ON dv.id = x.version_id
           WHERE x.image_id = i.id
      ) v ON true
      LEFT JOIN LATERAL (
          SELECT nl.class_id, nl.round_id FROM labels nl
           WHERE nl.image_id = i.id
             AND NOT EXISTS (SELECT 1 FROM dataset_version_labels y WHERE y.label_id = nl.id)
             AND nl.id = (SELECT max(id) FROM labels WHERE image_id = i.id)
      ) n ON true
      LEFT JOIN classes nc ON nc.id = n.class_id
      LEFT JOIN LATERAL (
          SELECT top1_class_id, confidence FROM captures
           WHERE image_id = i.id ORDER BY created_at DESC, id DESC LIMIT 1
      ) p ON true
      LEFT JOIN classes pc ON pc.id = p.top1_class_id
     WHERE i.dataset_id = %(dataset)s
     ORDER BY i.id
"""


def load_records(conn: psycopg.Connection, dataset: str) -> list[ImageRecord]:
    """Every image of the dataset; an unknown dataset has none."""
    row = conn.execute("SELECT id FROM datasets WHERE name = %s", (dataset,)).fetchone()
    if row is None:
        return []
    return [
        ImageRecord(
            image_id=r[0],
            s3_key=r[1],
            split=r[2],
            status=r[3],
            source=r[4],
            created_at=r[5],
            label=r[6],
            label_source=r[7],
            labeled_by=r[8],
            label_round=r[9],
            versions=r[10],
            new_label=r[11],
            new_label_round=r[12],
            app_prediction=r[13],
            app_confidence=r[14],
        )
        for r in conn.execute(_RECORDS, {"dataset": row[0]})
    ]


def dataset_names(conn: psycopg.Connection) -> list[str]:
    return [name for (name,) in conn.execute("SELECT name FROM datasets ORDER BY name")]
