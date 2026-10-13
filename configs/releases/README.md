# Model releases

One file per dataset, `configs/releases/{dataset}.yaml`, holding the model
the app downloads for that dataset (design doc F5):

```yaml
model: rocole-fastvit   # MLflow registered model name
version: 3              # MLflow model version, trained by a round of this dataset
```

Merging a change here releases that version: the `model-release` state
machine exports it to ONNX (the worker must be on) and records it in
`model_releases`. To roll back, revert the release commit; the earlier
version comes back as a `rollback` row. CI runs `al check-release` on every PR.
