import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from common import (
    CaptureRequest,
    CaptureResponse,
    DeviceConfig,
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


IMAGE = "ghcr.io/duytq5/duytq5-leaf-al-platform-worker:" + "b" * 40
CLASSES = [
    {"code": "healthy", "display_name": "Healthy"},
    {"code": "rust", "display_name": "Coffee leaf rust"},
]


def _job(type_: str, spec: dict, **extra) -> dict:
    return {
        "job_id": str(uuid4()),
        "type": type_,
        "image": IMAGE,
        "task_token": "tok",
        "spec": spec,
    } | extra


def _train() -> dict:
    spec = {
        "config": _config_dict(),
        "manifest_uri": "s3://al-data/manifests/durian/v7.parquet",
        "classes": CLASSES,
    }
    return _job("train", spec, round_id=str(uuid4()))


def _export() -> dict:
    spec = {
        "model_uri": "models:/leaf-disease/3",
        "output_prefix": "edge/leaf-disease/3/",
        "labels": CLASSES,
    }
    return _job("export", spec)


def test_parse_train_job_round_trip():
    job = parse_job(json.dumps(_train()))
    assert isinstance(job, TrainJob)
    assert [c.code for c in job.spec.classes] == ["healthy", "rust"]
    assert parse_job(job.model_dump_json()) == job


def test_parse_score_job_defaults_to_probs():
    spec = {
        "model_uri": "models:/leaf-disease/3",
        "pool_manifest_uri": "s3://x/p.parquet",
        "output_uri": "s3://x/scores/r/pool.parquet",
    }
    job = parse_job(json.dumps(_job("score", spec, round_id=str(uuid4()))))
    assert isinstance(job, ScoreJob)
    assert job.spec.outputs == {"probs"}


def test_score_job_requires_probs():
    spec = {
        "model_uri": "m",
        "pool_manifest_uri": "p",
        "output_uri": "o",
        "outputs": ["embeddings"],
    }
    with pytest.raises(ValidationError):
        parse_job(json.dumps(_job("score", spec, round_id=str(uuid4()))))


def test_parse_export_job():
    job = parse_job(json.dumps(_export()))
    assert isinstance(job, ExportJob)
    assert job.round_id is None
    assert job.spec.labels[1].display_name == "Coffee leaf rust"


@pytest.mark.parametrize("make", [_train, _export])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda j: j.pop("task_token"),  # every job resumes a state machine
        lambda j: j.pop("image"),
        lambda j: j.update(image=IMAGE.replace("b" * 40, "latest")),  # must pin a commit
        lambda j: j.update(image="docker.io/someone/worker:" + "b" * 40),
    ],
)
def test_jobs_need_a_task_token_and_a_pinned_worker_image(make, mutate):
    job = make()
    mutate(job)
    with pytest.raises(ValidationError):
        parse_job(json.dumps(job))


@pytest.mark.parametrize(
    "classes",
    [CLASSES[:1], [CLASSES[0], CLASSES[0]]],  # one class; duplicate codes
)
def test_job_class_lists_are_validated(classes):
    train, export = _train(), _export()
    train["spec"]["classes"] = classes
    export["spec"]["labels"] = classes
    for job in (train, export):
        with pytest.raises(ValidationError):
            parse_job(json.dumps(job))


def test_export_job_has_no_round():
    with pytest.raises(ValidationError):
        parse_job(json.dumps(_export() | {"round_id": str(uuid4())}))


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
            "probs": {"healthy": 0.31, "leaf_blight": 0.54, "algal_spot": 0.15},
        },
    }


def test_capture_request_from_doc_example():
    req = CaptureRequest.model_validate(_capture())
    assert req.inference.probs["leaf_blight"] == 0.54
    png = _capture()
    png["image"]["content_type"] = "image/png"
    CaptureRequest.model_validate(png)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(captured_at="2026-09-29T08:12:00"),  # no timezone
        lambda c: c["image"].update(sha256="XYZ"),
        lambda c: c["inference"].update(top1="leaf_blight"),  # computed by the server
        lambda c: c["inference"].update(confidence=0.54),  # computed by the server
        lambda c: c["inference"]["probs"].update(healthy=0.5),  # sum is 1.19
        lambda c: c["inference"].update(probs={"healthy": 1.0}),  # one class
        lambda c: c["inference"]["probs"].update({"Leaf Blight": 0.0}),  # not a class code
        lambda c: c["image"].update(size=10 * 1024 * 1024 + 1),
        lambda c: c["image"].update(content_type="image/heic"),
        lambda c: c.update(user_sub="someone"),  # comes from the JWT, not the body
    ],
)
def test_capture_request_rejects_bad_input(mutate):
    c = _capture()
    mutate(c)
    with pytest.raises(ValidationError):
        CaptureRequest.model_validate(c)


def test_capture_response_upload_matches_status():
    cid = str(uuid4())
    upload = {
        "url": "https://s3.example.com/put",
        "headers": {"Content-Type": "image/jpeg"},
        "expires_in": 900,
    }
    ok = {"capture_id": cid, "status": "upload_required", "upload": upload}
    CaptureResponse.model_validate(ok)
    CaptureResponse.model_validate({"capture_id": cid, "status": "already_exists"})
    with pytest.raises(ValidationError):
        CaptureResponse.model_validate({"capture_id": cid, "status": "upload_required"})
    with pytest.raises(ValidationError):
        CaptureResponse.model_validate(ok | {"status": "already_exists"})


def test_device_config_from_doc_example():
    doc = {
        "dataset": "durian",
        "model": "leaf-disease",
        "version": 13,
        "sha256": SHA,
        "manifest_url": "https://s3.example.com/manifest.json",
        "model_url": "https://s3.example.com/model.onnx",
        "expires_in": 900,
    }
    assert DeviceConfig.model_validate(doc).version == 13
    with pytest.raises(ValidationError):
        DeviceConfig.model_validate(doc | {"manifest": {}})


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
        "labels": [
            {"code": "healthy", "display_name": "Healthy"},
            {"code": "leaf_blight", "display_name": "Cháy lá"},
            {"code": "algal_spot", "display_name": "Algal spot"},
        ],
    }


def test_manifest_from_doc_example():
    m = ModelManifest.model_validate(_manifest())
    assert m.preprocess.input_size == (256, 256)


def test_manifest_rejects_duplicate_class_codes():
    m = _manifest()
    m["labels"] = [
        {"code": "healthy", "display_name": "A"},
        {"code": "healthy", "display_name": "B"},
    ]
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(m)


def test_s3_keys():
    assert s3keys.raw_key("durian", SHA) == f"raw/durian/{SHA}.jpg"
    assert s3keys.manifest_key("rocole", 3) == "manifests/rocole/v3.parquet"
    assert s3keys.edge_manifest_key("leaf-disease", 13) == "edge/leaf-disease/13/manifest.json"
