"""The current model of each dataset: its latest model_releases row (design
doc F5), so a rollback is just a newer row pointing at an older version.

A release whose ONNX columns are empty is never served (model-release records
the export before the release, so this should not happen); the app then keeps
the model it has.
"""

CURRENT_MODELS = """
    SELECT DISTINCT ON (mv.dataset_id)
           mv.dataset_id, mv.mlflow_name, mv.mlflow_version, mv.onnx_sha256
      FROM model_releases mr
      JOIN model_versions mv ON mv.id = mr.model_version_id
     ORDER BY mv.dataset_id, mr.released_at DESC, mr.id DESC
"""
