from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends

from api.deps import get_db
from api.releases import CURRENT_MODELS
from common.edge import CropsResponse

router = APIRouter()

_CROPS = f"""
    WITH current AS ({CURRENT_MODELS})
    SELECT c.code, c.display_name, c.description,
           d.name, d.display_name, cur.mlflow_version
      FROM crops c
      JOIN datasets d ON d.crop_id = c.id
      JOIN current cur ON cur.dataset_id = d.id
     WHERE cur.onnx_sha256 IS NOT NULL
     ORDER BY c.display_name, c.code, d.display_name, d.name
"""


@router.get("/v1/crops")
def crops(db: Annotated[psycopg.Connection, Depends(get_db)]) -> CropsResponse:
    """Public: the app home screen. Crops with no released dataset are left out."""
    out: dict[str, dict] = {}
    for code, crop_name, description, name, display_name, version in db.execute(_CROPS):
        crop = out.setdefault(
            code,
            {"code": code, "display_name": crop_name, "description": description, "datasets": []},
        )
        crop["datasets"].append({"name": name, "display_name": display_name, "version": version})
    return CropsResponse(crops=list(out.values()))
