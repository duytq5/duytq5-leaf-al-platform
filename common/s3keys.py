"""S3 key layout from the design doc, in one place.

s3://al-data/
  incoming/{sha256}.jpg
  raw/{dataset}/{sha256}.jpg
  manifests/{dataset}/v{n}.parquet
  scores/{round_id}/pool.parquet
  selections/{round_id}.json
s3://al-models/
  mlflow-artifacts/
  edge/{model}/{version}/model.onnx, manifest.json
"""

from uuid import UUID


def incoming_key(sha256: str) -> str:
    return f"incoming/{sha256}.jpg"


def raw_key(dataset: str, sha256: str) -> str:
    return f"raw/{dataset}/{sha256}.jpg"


def manifest_key(dataset: str, version: int) -> str:
    return f"manifests/{dataset}/v{version}.parquet"


def scores_key(round_id: UUID | str) -> str:
    return f"scores/{round_id}/pool.parquet"


def selection_key(round_id: UUID | str) -> str:
    return f"selections/{round_id}.json"


def edge_prefix(model: str, version: int) -> str:
    return f"edge/{model}/{version}/"


def edge_model_key(model: str, version: int) -> str:
    return edge_prefix(model, version) + "model.onnx"


def edge_manifest_key(model: str, version: int) -> str:
    return edge_prefix(model, version) + "manifest.json"
