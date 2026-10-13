"""The worker's jobs, run inside the worker image (design doc F4, F6, F5).

  train   downloads the version manifest's images (cached by sha256), trains
          the round's recipe on the train split, keeps the epoch with the best
          val macro F1, evaluates it on the test split, logs the model to
          MLflow and registers a version tagged with the job id.
  score   writes class probabilities for the round's unlabeled pool.
  export  writes edge/{model}/{version}/model.onnx and then manifest.json
          after checking ONNX and torch agree on the val split, and moves the
          MLflow aliases (champion-prev = old champion, champion = this one).

Each is idempotent on its job: a re-delivered train job finds its registered
version by job id, a score job finds its output, an export job finds its
manifest. Recipes never see test images; the platform evaluates on them.
"""

import contextlib
import copy
import hashlib
import io
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common.jobs import ExportJob, ScoreJob, ScoreResult, TrainJob, TrainResult
from common.manifest import ModelManifest
from common.release import ExportResult
from recipes import TrainRecipe, get_recipe
from recipes.base import TrainContext

log = logging.getLogger("worker.compute")

SSM_MODELS_BUCKET = "/leaf-al/buckets/models"
CHAMPION, CHAMPION_PREV = "champion", "champion-prev"
# Largest difference in a class probability allowed between ONNX and torch.
PARITY_ATOL = 1e-3


class Store(Protocol):
    def get(self, uri: str) -> bytes: ...
    def put(self, uri: str, body: bytes) -> None: ...
    def exists(self, uri: str) -> bool: ...


def split_uri(uri: str) -> tuple[str, str]:
    m = re.fullmatch(r"s3://([^/]+)/(.+)", uri)
    if not m:
        raise ValueError(f"not an s3:// URI: {uri}")
    return m.group(1), m.group(2)


class S3Store:
    def __init__(self, client) -> None:
        self.client = client

    def get(self, uri: str) -> bytes:
        bucket, key = split_uri(uri)
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()

    def put(self, uri: str, body: bytes) -> None:
        bucket, key = split_uri(uri)
        self.client.put_object(Bucket=bucket, Key=key, Body=body)

    def exists(self, uri: str) -> bool:
        from botocore.exceptions import ClientError

        bucket, key = split_uri(uri)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class WorkerEnv:
    store: Store
    cache_dir: Path
    models_bucket: str
    git_sha: str = "unknown"
    device: str = field(default_factory=_default_device)
    num_workers: int = 0

    @classmethod
    def from_environ(cls) -> "WorkerEnv":  # pragma: no cover - wiring only
        import boto3

        session = boto3.Session()
        bucket = (
            os.environ.get("MODELS_BUCKET")
            or session.client("ssm").get_parameter(Name=SSM_MODELS_BUCKET)["Parameter"]["Value"]
        )
        return cls(
            store=S3Store(session.client("s3")),
            cache_dir=Path(os.environ.get("WORKER_CACHE_DIR", "/cache")),
            models_bucket=bucket,
            git_sha=os.environ.get("GIT_SHA", "unknown"),
            num_workers=int(os.environ.get("WORKER_NUM_WORKERS", "4")),
        )


class ExportMismatch(RuntimeError):
    """The ONNX model does not give the torch model's predictions."""


# Data


def _read_parquet(store: Store, uri: str) -> dict[str, list]:
    return pq.read_table(io.BytesIO(store.get(uri))).to_pydict()


def _image_path(env: WorkerEnv, bucket: str, key: str, sha256: str) -> Path:
    """The image from the local cache, downloaded when missing or damaged."""
    path = env.cache_dir / "images" / f"{sha256}.jpg"
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
        return path
    body = env.store.get(f"s3://{bucket}/{key}")
    if hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError(f"s3://{bucket}/{key} does not match its sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return path


def _images(env: WorkerEnv, bucket: str, rows: dict[str, list], idx: list[int]) -> list[Path]:
    return [_image_path(env, bucket, rows["key"][i], rows["sha256"][i]) for i in idx]


class _Images:
    """torch Dataset over cached image files; label -1 when unknown."""

    def __init__(self, paths: list[Path], labels: list[int], transform) -> None:
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(self.paths[i]) as img:
            return self.transform(img.convert("RGB")), self.labels[i]


def _loader(env: WorkerEnv, data: _Images, batch_size: int, *, shuffle=False, seed=0):
    import torch

    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=env.num_workers,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        pin_memory=env.device == "cuda",
    )


def _autocast(env: WorkerEnv):
    import torch

    if env.device == "cuda":  # bf16 mixed precision on the RTX 5070
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


def _probs(env: WorkerEnv, model, loader) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad(), _autocast(env)():
        for images, _ in loader:
            out.append(torch.softmax(model(images.to(env.device)).float(), dim=1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0), dtype=np.float32)


def _metrics(prefix: str, probs: np.ndarray, labels: list[int], n: int) -> dict[str, float]:
    y = np.asarray(labels)
    pred = probs.argmax(axis=1)
    f1s = []
    for c in range(n):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return {
        f"{prefix}_accuracy": float(np.mean(pred == y)),
        f"{prefix}_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


# MLflow


def _model_version(model_uri: str):
    """The registered version a models:/name/N or models:/name@alias URI names."""
    from mlflow import MlflowClient

    m = re.fullmatch(r"models:/([^/@]+)(?:/(\d+)|@(.+))", model_uri)
    if not m:
        raise ValueError(f"not a registered model URI: {model_uri}")
    client = MlflowClient()
    if m.group(2):
        return client.get_model_version(m.group(1), m.group(2))
    return client.get_model_version_by_alias(m.group(1), m.group(3))


def _recipe_of(version) -> TrainRecipe:
    return get_recipe(version.tags["recipe"], json.loads(version.tags["recipe_params"]))


def _load(model_uri: str, device: str):
    import mlflow

    return mlflow.pytorch.load_model(model_uri, map_location=device)


# Jobs


def run_train(job: TrainJob, env: WorkerEnv) -> dict:
    import mlflow
    import torch
    from mlflow import MlflowClient

    spec = job.spec
    recipe = get_recipe(spec.recipe, spec.params)
    name = recipe.model_name(spec.dataset)
    client = MlflowClient()
    done = _registered_for(client, name, job)
    if done is not None:
        log.info("job %s already registered %s version %s", job.job_id, name, done.version)
        return _train_result(client, recipe, done)

    rows = _read_parquet(env.store, spec.manifest_uri)
    codes = [c.code for c in spec.classes]
    unknown = sorted(set(rows["class_code"]) - set(codes))
    if unknown:
        raise ValueError(f"the manifest has classes the job does not list: {unknown}")
    labels = [codes.index(c) for c in rows["class_code"]]
    bucket, _ = split_uri(spec.manifest_uri)
    splits = {s: [i for i, x in enumerate(rows["split"]) if x == s] for s in ("train", "val")}
    test = [i for i, x in enumerate(rows["split"]) if x == "test"]
    if not splits["train"] or not splits["val"]:
        raise ValueError("the manifest needs train and val images")

    p = recipe.params
    torch.manual_seed(p.seed)

    def data(idx: list[int], transform) -> _Images:
        return _Images(_images(env, bucket, rows, idx), [labels[i] for i in idx], transform)

    train = _loader(
        env, data(splits["train"], recipe.train_transforms()), p.batch_size, shuffle=True,
        seed=p.seed,
    )  # fmt: skip
    val = _loader(env, data(splits["val"], recipe.eval_transforms()), p.batch_size)
    val_labels = [labels[i] for i in splits["val"]]

    base = _load(spec.base_model, env.device) if spec.base_model else None
    model = recipe.build_model(len(codes), base).to(env.device)

    mlflow.set_experiment(f"leaf-al-{spec.dataset}")
    with mlflow.start_run(run_name=f"round-{job.round_id}") as run:
        mlflow.set_tags(_tags(job, recipe, env))
        mlflow.log_params({"recipe": spec.recipe, **p.model_dump(mode="json")})
        best: dict = {"f1": -1.0}

        def end_epoch(epoch: int, m, train_loss: float) -> None:
            scores = _metrics("val", _probs(env, m, val), val_labels, len(codes))
            mlflow.log_metrics({"train_loss": train_loss, **scores}, step=epoch)
            if scores["val_macro_f1"] > best["f1"]:
                best.update(f1=scores["val_macro_f1"], epoch=epoch, metrics=scores)
                best["state"] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

        ctx = TrainContext(
            train_loader=train, device=env.device, autocast=_autocast(env), end_epoch=end_epoch
        )
        recipe.fit(model, ctx)
        if "state" not in best:
            raise RuntimeError("the recipe finished without ending an epoch")
        model.load_state_dict(best["state"])

        metrics = dict(best["metrics"], best_epoch=float(best["epoch"]))
        if test:  # the platform, not the recipe, looks at the test split
            test_loader = _loader(env, data(test, recipe.eval_transforms()), p.batch_size)
            metrics |= _metrics(
                "test", _probs(env, model, test_loader), [labels[i] for i in test], len(codes)
            )
        mlflow.log_metrics(metrics)
        info = mlflow.pytorch.log_model(
            model.cpu().eval(), name="model", serialization_format="pickle", pip_requirements=[]
        )
        version = mlflow.register_model(info.model_uri, name, tags=_tags(job, recipe, env))
        run_id = run.info.run_id
    log.info("registered %s version %s (run %s)", name, version.version, run_id)
    return _train_result(client, recipe, client.get_model_version(name, version.version))


def _tags(job: TrainJob, recipe: TrainRecipe, env: WorkerEnv) -> dict[str, str]:
    spec = job.spec
    tags = {
        "job_id": str(job.job_id),
        "round_id": str(job.round_id),
        "dataset": spec.dataset,
        "manifest_uri": spec.manifest_uri,
        "recipe": spec.recipe,
        "recipe_params": recipe.params.model_dump_json(),
        "git_sha": env.git_sha,
    }
    if spec.base_model:
        tags["base_model"] = spec.base_model
    return tags


def _registered_for(client, name: str, job: TrainJob):
    from mlflow.exceptions import MlflowException

    try:
        found = client.search_model_versions(f"name='{name}' AND tags.job_id='{job.job_id}'")
    except MlflowException:  # no such registered model yet
        return None
    return min(found, key=lambda v: int(v.version)) if found else None


def _train_result(client, recipe: TrainRecipe, version) -> dict:
    run = client.get_run(version.run_id)
    keep = ("val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1")
    return TrainResult(
        mlflow_name=version.name,
        mlflow_version=int(version.version),
        mlflow_run_id=version.run_id,
        model_uri=f"models:/{version.name}/{version.version}",
        arch=recipe.arch,
        metrics={k: v for k, v in run.data.metrics.items() if k in keep},
    ).model_dump(mode="json")


def run_score(job: ScoreJob, env: WorkerEnv) -> dict:
    spec = job.spec
    if "embeddings" in spec.outputs:
        raise ValueError("this worker writes probs only; no strategy needs embeddings yet")
    if env.store.exists(spec.output_uri):
        n = pq.read_table(io.BytesIO(env.store.get(spec.output_uri))).num_rows
        return ScoreResult(output_uri=spec.output_uri, n_images=n).model_dump(mode="json")

    recipe = _recipe_of(_model_version(spec.model_uri))
    model = _load(spec.model_uri, env.device)
    rows = _read_parquet(env.store, spec.pool_manifest_uri)
    bucket, _ = split_uri(spec.pool_manifest_uri)
    idx = list(range(len(rows["image_id"])))
    data = _Images(_images(env, bucket, rows, idx), [-1] * len(idx), recipe.eval_transforms())
    probs = _probs(env, model, _loader(env, data, recipe.params.batch_size))
    table = pa.table(
        {
            "image_id": pa.array(rows["image_id"], type=pa.int64()),
            "probs": pa.array([list(map(float, r)) for r in probs], type=pa.list_(pa.float32())),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    env.store.put(spec.output_uri, buf.getvalue())
    return ScoreResult(output_uri=spec.output_uri, n_images=len(idx)).model_dump(mode="json")


def run_export(job: ExportJob, env: WorkerEnv) -> dict:
    spec = job.spec
    version = _model_version(spec.model_uri)
    name, number = version.name, int(version.version)
    prefix = f"s3://{env.models_bucket}/{spec.output_prefix}"
    onnx_key = spec.output_prefix + "model.onnx"

    if env.store.exists(prefix + "manifest.json"):  # exported before (re-run or rollback)
        manifest = ModelManifest.model_validate_json(env.store.get(prefix + "manifest.json"))
    else:
        recipe = _recipe_of(version)
        model = recipe.export_model(_load(spec.model_uri, "cpu").eval())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            _export_onnx(model, recipe, path)
            _check_parity(env, model, recipe, version.tags["manifest_uri"], path)
            body = path.read_bytes()
        manifest = ModelManifest(
            model=name,
            version=number,
            sha256=hashlib.sha256(body).hexdigest(),
            size_bytes=len(body),
            preprocess=recipe.preprocess(),
            labels=spec.labels,
        )
        env.store.put(prefix + "model.onnx", body)
        env.store.put(prefix + "manifest.json", manifest.model_dump_json().encode())
    _move_champion(name, number)
    return ExportResult(
        onnx_s3_key=onnx_key, onnx_sha256=manifest.sha256, size_bytes=manifest.size_bytes
    ).model_dump(mode="json")


def _export_onnx(model, recipe: TrainRecipe, path: Path) -> None:
    import torch

    size = recipe.preprocess().input_size
    example = torch.zeros(2, 3, *size)
    program = torch.onnx.export(
        copy.deepcopy(model),
        (example,),
        dynamo=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        verbose=False,
    )
    program.save(str(path), external_data=False)


def _onnx_probs(path: Path, images: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (logits,) = session.run(None, {session.get_inputs()[0].name: images})
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _check_parity(env: WorkerEnv, model, recipe: TrainRecipe, manifest_uri: str, path: Path):
    """ONNX and torch must agree on every val image (design doc F5 step 3)."""
    import torch

    rows = _read_parquet(env.store, manifest_uri)
    bucket, _ = split_uri(manifest_uri)
    idx = [i for i, s in enumerate(rows["split"]) if s == "val"]
    if not idx:
        raise ExportMismatch("no val images to check the ONNX model on")
    data = _Images(_images(env, bucket, rows, idx), [-1] * len(idx), recipe.eval_transforms())
    worst, disagree = 0.0, 0
    for images, _ in _loader(env, data, recipe.params.batch_size):
        with torch.no_grad():
            ref = torch.softmax(model(images).float(), dim=1).numpy()
        got = _onnx_probs(path, images.numpy())
        worst = max(worst, float(np.abs(got - ref).max()))
        disagree += int(np.sum(got.argmax(axis=1) != ref.argmax(axis=1)))
    if worst > PARITY_ATOL or disagree:
        raise ExportMismatch(
            f"ONNX and torch disagree on val: max probability difference {worst:.2g}"
            f" (allowed {PARITY_ATOL}), {disagree} of {len(idx)} predictions differ"
        )


def _move_champion(name: str, version: int) -> None:
    """champion = version; champion-prev = the champion it replaces. A re-run
    for the current champion changes nothing."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    try:
        current = int(client.get_model_version_by_alias(name, CHAMPION).version)
    except MlflowException:
        current = None
    if current == version:
        return
    if current is not None:
        client.set_registered_model_alias(name, CHAMPION_PREV, str(current))
    client.set_registered_model_alias(name, CHAMPION, str(version))
