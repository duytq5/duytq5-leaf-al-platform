import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from common import (
    CaptureRequest,
    CaptureResponse,
    ExportJob,
    ModelManifest,
    ScoreJob,
    TrainConfig,
    TrainJob,
    parse_job,
    s3keys,
)

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def _config_dict() -> dict:
    return TrainConfig.from_yaml(ROOT / "configs/train/example.yaml").model_dump(mode="json")


def test_example_config_loads():
    cfg = TrainConfig.from_yaml(ROOT / "configs/train/example.yaml")
    assert cfg.model.arch == "fastvit_t8"
    assert cfg.train.lr == pytest.approx(3e-4)  # YAML gives "3e-4" as a string
    assert cfg.base_model is None


def test_config_rejects_unknown_keys():
    bad = _config_dict() | {"optimizer": "sgd"}
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(bad)


def test_parse_train_job_round_trip():
    msg = {
        "job_id": str(uuid4()),
        "type": "train",
        "task_token": "tok",
        "round_id": str(uuid4()),
        "spec": {
            "config": _config_dict(),
            "manifest_uri": "s3://al-data/manifests/durian/v7.parquet",
        },
    }
    job = parse_job(json.dumps(msg))
    assert isinstance(job, TrainJob)
    assert parse_job(job.model_dump_json()) == job


def test_parse_score_job_defaults_to_probs():
    msg = {
        "job_id": str(uuid4()),
        "type": "score",
        "task_token": "tok",
        "round_id": str(uuid4()),
        "spec": {
            "model_uri": "models:/leaf-disease/3",
            "pool_manifest_uri": "s3://x/p.parquet",
            "output_uri": "s3://x/scores/r/pool.parquet",
        },
    }
    job = parse_job(json.dumps(msg))
    assert isinstance(job, ScoreJob)
    assert job.spec.outputs == {"probs"}


def test_score_job_requires_probs():
    msg = {
        "job_id": str(uuid4()),
        "type": "score",
        "task_token": "tok",
        "round_id": str(uuid4()),
        "spec": {
            "model_uri": "m",
            "pool_manifest_uri": "p",
            "output_uri": "o",
            "outputs": ["embeddings"],
        },
    }
    with pytest.raises(ValidationError):
        parse_job(json.dumps(msg))


def test_train_job_requires_task_token_but_export_does_not():
    train = {
        "job_id": str(uuid4()),
        "type": "train",
        "round_id": str(uuid4()),
        "spec": {"config": _config_dict(), "manifest_uri": "s3://x"},
    }
    with pytest.raises(ValidationError):
        parse_job(json.dumps(train))
    export = {
        "job_id": str(uuid4()),
        "type": "export",
        "spec": {"model_uri": "models:/leaf-disease/3", "output_prefix": "edge/leaf-disease/3/"},
    }
    assert isinstance(parse_job(json.dumps(export)), ExportJob)


def test_unknown_job_type_rejected():
    with pytest.raises(ValidationError):
        parse_job(json.dumps({"job_id": str(uuid4()), "type": "evaluate", "spec": {}}))


def _capture() -> dict:
    return {
        "capture_id": str(uuid4()),
        "dataset": "durian",
        "captured_at": "2026-09-29T08:12:00+07:00",
        "image": {"sha256": SHA, "content_type": "image/jpeg", "size": 842113},
        "inference": {
            "model_version": "leaf-disease/12",
            "top1": "leaf_blight",
            "confidence": 0.54,
            "probs": {"healthy": 0.31, "leaf_blight": 0.54, "algal_spot": 0.15},
        },
    }


def test_capture_request_from_doc_example():
    req = CaptureRequest.model_validate(_capture())
    assert req.inference.top1 == "leaf_blight"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(captured_at="2026-09-29T08:12:00"),  # no timezone
        lambda c: c["image"].update(sha256="XYZ"),
        lambda c: c["inference"].update(top1="rust"),  # not in probs
        lambda c: c["inference"].update(confidence=1.5),
        lambda c: c.update(device_id="phone-1"),  # comes from the JWT, not the body
    ],
)
def test_capture_request_rejects_bad_input(mutate):
    c = _capture()
    mutate(c)
    with pytest.raises(ValidationError):
        CaptureRequest.model_validate(c)


def test_capture_response_upload_matches_status():
    cid = str(uuid4())
    upload = {"url": "https://s3.example.com/put", "expires_in": 900}
    ok = {"capture_id": cid, "status": "upload_required", "upload": upload}
    CaptureResponse.model_validate(ok)
    CaptureResponse.model_validate({"capture_id": cid, "status": "already_exists"})
    with pytest.raises(ValidationError):
        CaptureResponse.model_validate({"capture_id": cid, "status": "upload_required"})
    with pytest.raises(ValidationError):
        CaptureResponse.model_validate(ok | {"status": "already_exists"})


def _manifest() -> dict:
    return {
        "model": "leaf-disease",
        "version": 13,
        "format": "onnx",
        "sha256": SHA,
        "size_bytes": 14200000,
        "preprocess": {
            "input_size": [256, 256],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "labels": ["healthy", "leaf_blight", "algal_spot"],
    }


def test_manifest_from_doc_example():
    m = ModelManifest.model_validate(_manifest())
    assert m.preprocess.input_size == (256, 256)


def test_manifest_rejects_duplicate_labels():
    m = _manifest()
    m["labels"] = ["healthy", "healthy"]
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(m)


def test_s3_keys():
    assert s3keys.raw_key("durian", SHA) == f"raw/durian/{SHA}.jpg"
    assert s3keys.manifest_key("rocole", 3) == "manifests/rocole/v3.parquet"
    assert s3keys.edge_manifest_key("leaf-disease", 13) == "edge/leaf-disease/13/manifest.json"
