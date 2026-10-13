"""The worker's train, score and export jobs (design doc F4, F6, F5), run
for real on the CPU with a tiny timm model, an in-memory S3 and a SQLite
MLflow. Skipped without the `worker` extra (torch, timm, mlflow, onnx).
"""

import hashlib
import io
import json
import uuid

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("mlflow")
pytest.importorskip("onnxruntime")

import mlflow
import pyarrow as pa
import pyarrow.parquet as pq
from mlflow import MlflowClient
from PIL import Image

from common.jobs import (
    ExportJob,
    ScoreJob,
    ScoreResult,
    TrainJob,
    TrainResult,
)
from common.manifest import ModelManifest
from common.release import ExportResult
from lambdas.round import read_scores
from worker import compute

COMMIT = "c" * 40
IMAGE = f"ghcr.io/duytq5/duytq5-leaf-al-platform-worker:{COMMIT}"
CLASSES = [
    {"code": "red", "display_name": "Red"},
    {"code": "green", "display_name": "Green"},
    {"code": "blue", "display_name": "Blue"},
]
COLORS = {"red": (200, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 200)}
TINY = {
    "arch": "test_efficientnet",
    "pretrained": False,
    "input_size": 32,
    "epochs": 2,
    "batch_size": 4,
    "lr": 1e-3,
}
MANIFEST = "s3://data/manifests/toy/v0.parquet"
POOL = "s3://data/scores/r/pool-manifest.parquet"
SCORES = "s3://data/scores/r/pool.parquet"


class FakeS3:
    """In-memory S3 with the worker's store interface; records every call."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.puts: list[str] = []

    def get(self, uri: str) -> bytes:
        self.gets.append(uri)
        return self.objects[uri]

    def put(self, uri: str, body: bytes) -> None:
        self.puts.append(uri)
        self.objects[uri] = body

    def exists(self, uri: str) -> bool:
        return uri in self.objects


def _jpeg(color, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = np.clip(np.array(color) + rng.normal(0, 20, (40, 40, 3)), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _parquet(columns: dict) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.table(columns), buf)
    return buf.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow's local artifact root
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_registry_uri(f"sqlite:///{tmp_path}/mlflow.db")
    s3 = FakeS3()
    rows = {"image_id": [], "sha256": [], "key": [], "split": [], "class_code": []}
    pool = {"image_id": [], "sha256": [], "key": []}
    n = 0
    for code, color in COLORS.items():
        for split, count in (("train", 6), ("val", 3), ("test", 3), ("pool", 2)):
            for _ in range(count):
                n += 1
                body = _jpeg(color, n)
                sha = hashlib.sha256(body).hexdigest()
                key = f"raw/toy/{sha}.jpg"
                s3.objects[f"s3://data/{key}"] = body
                target = pool if split == "pool" else rows
                target["image_id"].append(n)
                target["sha256"].append(sha)
                target["key"].append(key)
                if split != "pool":
                    rows["split"].append(split)
                    rows["class_code"].append(code)
    s3.objects[MANIFEST] = _parquet(rows)
    s3.objects[POOL] = _parquet(pool)
    yield compute.WorkerEnv(
        store=s3, cache_dir=tmp_path / "cache", models_bucket="models", git_sha=COMMIT
    )
    mlflow.set_tracking_uri(None)


def _train_job(params=None, base_model=None, job_id=None) -> TrainJob:
    return TrainJob(
        job_id=job_id or uuid.uuid4(),
        image=IMAGE,
        task_token="t",
        round_id=uuid.uuid4(),
        spec={
            "recipe": "timm_classifier",
            "params": params or TINY,
            "base_model": base_model,
            "dataset": "toy",
            "manifest_uri": MANIFEST,
            "classes": CLASSES,
        },
    )


def _score_job(model_uri: str) -> ScoreJob:
    return ScoreJob(
        job_id=uuid.uuid4(),
        image=IMAGE,
        task_token="t",
        round_id=uuid.uuid4(),
        spec={"model_uri": model_uri, "pool_manifest_uri": POOL, "output_uri": SCORES},
    )


def _export_job(version: int) -> ExportJob:
    return ExportJob(
        job_id=uuid.uuid4(),
        image=IMAGE,
        task_token="t",
        spec={
            "model_uri": f"models:/toy-test/{version}",
            "output_prefix": f"edge/toy-test/{version}/",
            "labels": CLASSES,
        },
    )


def _alias(alias: str) -> int:
    return int(MlflowClient().get_model_version_by_alias("toy-test", alias).version)


# train


def test_train_registers_a_version_with_its_metrics(env):
    job = _train_job()
    result = TrainResult.model_validate(compute.run_train(job, env))
    # Named after the dataset and the model family (test_efficientnet -> test).
    assert (result.mlflow_name, result.mlflow_version) == ("toy-test", 1)
    assert result.model_uri == "models:/toy-test/1"
    assert result.arch == "test_efficientnet"
    assert {"val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"} <= set(result.metrics)
    assert all(0.0 <= v <= 1.0 for v in result.metrics.values())

    version = MlflowClient().get_model_version("toy-test", "1")
    assert version.run_id == result.mlflow_run_id
    tags = version.tags
    assert tags["job_id"] == str(job.job_id) and tags["round_id"] == str(job.round_id)
    assert tags["recipe"] == "timm_classifier" and tags["git_sha"] == COMMIT
    assert tags["manifest_uri"] == MANIFEST
    assert json.loads(tags["recipe_params"])["arch"] == "test_efficientnet"
    run = MlflowClient().get_run(result.mlflow_run_id)
    assert run.data.params["epochs"] == "2"


def test_train_is_idempotent_on_job_id(env):
    """SQS can deliver the same job twice: the second run trains nothing."""
    job = _train_job()
    first = compute.run_train(job, env)
    assert compute.run_train(job, env) == first
    assert len(MlflowClient().search_model_versions("name='toy-test'")) == 1


def test_images_are_cached_by_sha256(env):
    compute.run_train(_train_job(), env)
    env.store.gets.clear()
    compute.run_train(_train_job(), env)
    assert not [u for u in env.store.gets if "/raw/" in u]


def test_a_cached_image_with_the_wrong_bytes_is_downloaded_again(env):
    compute.run_train(_train_job(), env)
    for path in (env.cache_dir / "images").iterdir():
        path.write_bytes(b"corrupt")
    env.store.gets.clear()
    compute.run_train(_train_job(), env)
    assert [u for u in env.store.gets if "/raw/" in u]


def test_the_model_never_trains_on_test_images(env, monkeypatch):
    from recipes.base import TrainRecipe

    seen = []
    original = TrainRecipe.train_step

    def spy(self, model, batch, optimizer, loss_fn):
        seen.append(len(batch[1]))
        return original(self, model, batch, optimizer, loss_fn)

    monkeypatch.setattr(TrainRecipe, "train_step", spy)
    compute.run_train(_train_job(), env)
    assert sum(seen) == 2 * 18  # two epochs over the 18 train images only


def test_fine_tune_from_a_registered_model(env):
    compute.run_train(_train_job(), env)
    result = compute.run_train(_train_job(base_model="models:/toy-test/1"), env)
    assert result["mlflow_version"] == 2
    assert MlflowClient().get_model_version("toy-test", "2").tags["base_model"] == (
        "models:/toy-test/1"
    )


def test_an_unknown_class_in_the_manifest_fails(env):
    job = _train_job()
    job.spec.classes = job.spec.classes[:2]
    with pytest.raises(ValueError, match="blue"):
        compute.run_train(job, env)


# score


def test_score_writes_probabilities_for_the_pool(env):
    compute.run_train(_train_job(), env)
    result = ScoreResult.model_validate(compute.run_score(_score_job("models:/toy-test/1"), env))
    assert result == ScoreResult(output_uri=SCORES, n_images=6)
    scores = read_scores(env.store.objects[SCORES])
    pool = pq.read_table(io.BytesIO(env.store.objects[POOL]))
    assert scores.ids == [str(i) for i in pool.column("image_id").to_pylist()]
    assert scores.probs.shape == (6, 3)
    assert np.allclose(scores.probs.sum(axis=1), 1.0, atol=1e-4)


def test_score_is_idempotent(env):
    compute.run_train(_train_job(), env)
    job = _score_job("models:/toy-test/1")
    first = compute.run_score(job, env)
    env.store.puts.clear()
    assert compute.run_score(job, env) == first
    assert env.store.puts == []


def test_score_refuses_embeddings(env):
    job = _score_job("models:/toy-test/1")
    job.spec.outputs = frozenset({"probs", "embeddings"})
    with pytest.raises(ValueError, match="embeddings"):
        compute.run_score(job, env)


# export


def test_export_writes_the_model_then_the_manifest_and_moves_champion(env):
    compute.run_train(_train_job(), env)
    result = ExportResult.model_validate(compute.run_export(_export_job(1), env))
    onnx_uri = "s3://models/edge/toy-test/1/model.onnx"
    manifest_uri = "s3://models/edge/toy-test/1/manifest.json"
    assert env.store.puts == [onnx_uri, manifest_uri]  # manifest last
    body = env.store.objects[onnx_uri]
    assert result == ExportResult(
        onnx_s3_key="edge/toy-test/1/model.onnx",
        onnx_sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
    )
    manifest = ModelManifest.model_validate_json(env.store.objects[manifest_uri])
    assert (manifest.model, manifest.version, manifest.sha256) == (
        "toy-test",
        1,
        result.onnx_sha256,
    )
    assert manifest.preprocess.input_size == (32, 32)
    assert [c.code for c in manifest.labels] == ["red", "green", "blue"]
    assert _alias("champion") == 1


def test_the_onnx_model_runs_like_the_torch_model(env):
    import onnxruntime as ort

    compute.run_train(_train_job(), env)
    compute.run_export(_export_job(1), env)
    session = ort.InferenceSession(env.store.objects["s3://models/edge/toy-test/1/model.onnx"])
    x = np.random.default_rng(0).random((2, 3, 32, 32), dtype=np.float32)
    (logits,) = session.run(None, {session.get_inputs()[0].name: x})
    assert logits.shape == (2, 3)


def test_release_promote_rerun_and_rollback(env):
    compute.run_train(_train_job(), env)
    compute.run_train(_train_job(), env)
    compute.run_export(_export_job(1), env)
    compute.run_export(_export_job(2), env)
    assert _alias("champion") == 2
    assert _alias("champion-prev") == 1

    env.store.puts.clear()
    again = compute.run_export(_export_job(2), env)  # a re-delivered job
    assert env.store.puts == []
    assert _alias("champion-prev") == 1

    back = compute.run_export(_export_job(1), env)  # rollback: no new export
    assert env.store.puts == []
    assert back["onnx_s3_key"] == "edge/toy-test/1/model.onnx" != again["onnx_s3_key"]
    assert _alias("champion") == 1
    assert _alias("champion-prev") == 2


def test_export_fails_when_onnx_and_torch_disagree(env, monkeypatch):
    compute.run_train(_train_job(), env)
    real = compute._onnx_probs
    monkeypatch.setattr(compute, "_onnx_probs", lambda *a: real(*a)[:, ::-1])
    with pytest.raises(compute.ExportMismatch):
        compute.run_export(_export_job(1), env)
    assert not [u for u in env.store.puts if "/edge/" in u]
    assert not MlflowClient().get_registered_model("toy-test").aliases
