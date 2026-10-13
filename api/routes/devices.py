from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import URL_EXPIRES_IN, Settings, get_db, get_s3, get_settings, presign_get
from api.releases import CURRENT_MODELS
from common.edge import DeviceConfig
from common.s3keys import edge_manifest_key, edge_model_key
from common.types import DatasetName

router = APIRouter()

_CURRENT = f"""
    WITH current AS ({CURRENT_MODELS})
    SELECT cur.mlflow_name, cur.mlflow_version, cur.onnx_sha256
      FROM datasets d
      LEFT JOIN current cur ON cur.dataset_id = d.id
     WHERE d.name = %s
"""


@router.get("/v1/devices/config")
def device_config(
    dataset: Annotated[DatasetName, Query()],
    db: Annotated[psycopg.Connection, Depends(get_db)],
    s3: Annotated[object, Depends(get_s3)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DeviceConfig:
    """Public: the current model of one dataset and links to download it.
    404 when there is none yet; the app then keeps its bundled model."""
    row = db.execute(_CURRENT, (dataset,)).fetchone()
    if row is None:
        raise HTTPException(404, f"No dataset named {dataset}.")
    model, version, sha256 = row
    if sha256 is None:
        raise HTTPException(404, f"No model has been released for {dataset} yet.")
    bucket = settings.models_bucket
    return DeviceConfig(
        dataset=dataset,
        model=model,
        version=version,
        sha256=sha256,
        model_url=presign_get(s3, bucket, edge_model_key(model, version)),
        manifest_url=presign_get(s3, bucket, edge_manifest_key(model, version)),
        expires_in=URL_EXPIRES_IN,
    )
