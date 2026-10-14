"""Image viewer (FiftyOne on the EC2 host): what it reads from Postgres, how it
copies images from S3, and the samples it shows.

FiftyOne itself is only in the viewer image (deploy/ec2/viewer), so the sample
tests are skipped where it is not installed.
"""

import uuid
from datetime import UTC, datetime

import pytest
from botocore.stub import Stubber

from tests.apidb import DATA_BUCKET, crop, dataset, fake_s3
from viewer.media import sync_media
from viewer.records import ImageRecord, load_records

COMMIT = "e" * 40


def _image(db, dataset_id, n, *, split="train", status="unlabeled", source="seed") -> int:
    return db.execute(
        "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
        " VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (dataset_id, f"{n:064x}", f"raw/durian/{n:064x}.jpg", status, split, source),
    ).fetchone()[0]


def _class(db, dataset_id, code) -> int:
    return db.execute(
        "SELECT id FROM classes WHERE dataset_id = %s AND code = %s", (dataset_id, code)
    ).fetchone()[0]


def _label(db, image_id, class_id, round_id=None) -> int:
    source, by = ("expert", "ls-7") if round_id else ("seed", None)
    return db.execute(
        "INSERT INTO labels (image_id, round_id, class_id, source, labeled_by)"
        " VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (image_id, round_id, class_id, source, by),
    ).fetchone()[0]


def _version_id(db, dataset_id, n) -> int:
    return db.execute(
        "SELECT id FROM dataset_versions WHERE dataset_id = %s AND version = %s", (dataset_id, n)
    ).fetchone()[0]


def _in_version(db, version_id, image_id, label_id) -> None:
    db.execute(
        "INSERT INTO dataset_version_labels (version_id, image_id, label_id) VALUES (%s, %s, %s)",
        (version_id, image_id, label_id),
    )


def _round(db, dataset_id, status) -> uuid.UUID:
    round_id = uuid.uuid4()
    db.execute(
        "INSERT INTO rounds (id, dataset_id, base_version_id, strategy, k, seed, status,"
        " started_by, commit_sha) VALUES (%s, %s, %s, 'margin', 2, 0, %s, 'duytq5', %s)",
        (round_id, dataset_id, _version_id(db, dataset_id, 0), status, COMMIT),
    )
    return round_id


@pytest.fixture
def durian(db):
    """v0: image 1 (seed a). Round 1 labels image 2 as b -> v1. Round 2 (running)
    has labeled image 3 as a (not merged yet) and still waits on image 4.
    Image 5 is an app upload in the pool with the app's guess; 6 was rejected;
    7 is a val image."""
    d = dataset(db, "durian", crop(db, "durian", "Sầu riêng"), "Lá sầu riêng")
    a, b = _class(db, d, "a"), _class(db, d, "b")
    v0 = _version_id(db, d, 0)
    ids = {n: _image(db, d, n, status="labeled") for n in (1, 2, 3)}
    ids[4] = _image(db, d, 4, status="queued")
    ids[5] = _image(db, d, 5, source="edge")
    ids[6] = _image(db, d, 6, status="rejected", source="edge")
    ids[7] = _image(db, d, 7, split="val", status="labeled")
    _in_version(db, v0, ids[1], _label(db, ids[1], a))
    _in_version(db, v0, ids[7], _label(db, ids[7], b))
    r1 = _round(db, d, "succeeded")
    v1 = db.execute(
        "INSERT INTO dataset_versions (dataset_id, version, parent_version_id,"
        " created_by_round_id) VALUES (%s, 1, %s, %s) RETURNING id",
        (d, v0, r1),
    ).fetchone()[0]
    db.execute(
        "INSERT INTO dataset_version_labels (version_id, image_id, label_id)"
        " SELECT %s, image_id, label_id FROM dataset_version_labels WHERE version_id = %s",
        (v1, v0),
    )
    _in_version(db, v1, ids[2], _label(db, ids[2], b, r1))
    r2 = _round(db, d, "running")
    _label(db, ids[3], a, r2)
    # The app's guess for image 5, from the capture it came with.
    model = db.execute(
        "INSERT INTO model_versions (round_id, trained_on_version_id, mlflow_name,"
        " mlflow_version, mlflow_run_id, arch, train_config, labels)"
        " VALUES (%s, %s, 'durian-fastvit', 1, 'run', 'fastvit_t8', '{}',"
        """ '[{"code": "a", "display_name": "A"}, {"code": "b", "display_name": "B"}]')"""
        " RETURNING id",
        (r1, v0),
    ).fetchone()[0]
    db.execute(
        "INSERT INTO captures (capture_id, image_id, dataset_id, sha256, user_sub,"
        " model_version_id, top1_class_id, confidence, probs, captured_at, content_type,"
        " size_bytes)"
        """ VALUES (%s, %s, %s, %s, 'u-1', %s, %s, 0.75, '{"a": 0.25, "b": 0.75}', now(),"""
        " 'image/jpeg', 1000)",
        (uuid.uuid4(), ids[5], d, f"{5:064x}", model, b),
    )
    return {"ids": ids, "rounds": (r1, r2)}


def test_records_of_every_image(db, durian):
    ids, (r1, r2) = durian["ids"], durian["rounds"]
    records = {r.image_id: r for r in load_records(db, "durian")}
    assert set(records) == set(ids.values())

    seed = records[ids[1]]
    assert (seed.split, seed.status, seed.source) == ("train", "labeled", "seed")
    assert (seed.label, seed.label_source, seed.label_round) == ("a", "seed", None)
    assert seed.versions == [0, 1]
    assert seed.new_label is None

    merged = records[ids[2]]
    assert (merged.label, merged.label_source, merged.labeled_by) == ("b", "expert", "ls-7")
    assert merged.label_round == r1
    assert merged.versions == [1]

    # Labeled in the running round, not in a version yet.
    fresh = records[ids[3]]
    assert fresh.label is None and fresh.versions == []
    assert (fresh.new_label, fresh.new_label_round) == ("a", r2)

    queued = records[ids[4]]
    assert (queued.status, queued.label, queued.new_label) == ("queued", None, None)

    upload = records[ids[5]]
    assert (upload.source, upload.status) == ("edge", "unlabeled")
    assert (upload.app_prediction, upload.app_confidence) == ("b", pytest.approx(0.75))

    assert records[ids[6]].status == "rejected"
    assert (records[ids[7]].split, records[ids[7]].label) == ("val", "b")
    assert records[ids[1]].s3_key == f"raw/durian/{1:064x}.jpg"
    assert isinstance(seed.created_at, datetime) and seed.created_at.tzinfo


def test_records_of_other_datasets_are_not_read(db, durian):
    other = dataset(db, "rocole", crop(db, "coffee", "Cà phê"), "Lá cà phê")
    _image(db, other, 99)
    assert len(load_records(db, "durian")) == 7
    assert [r.s3_key for r in load_records(db, "rocole")] == [f"raw/durian/{99:064x}.jpg"]


# S3 -> local files


def test_sync_media_downloads_only_missing_files(tmp_path):
    present = f"raw/durian/{1:064x}.jpg"
    missing = f"raw/durian/{2:064x}.jpg"
    (tmp_path / present).parent.mkdir(parents=True)
    (tmp_path / present).write_bytes(b"old")
    s3 = fake_s3()
    with Stubber(s3) as stub:
        stub.add_response(
            "get_object",
            {"Body": _Body(b"jpeg")},
            {"Bucket": DATA_BUCKET, "Key": missing},
        )
        paths = sync_media(s3, DATA_BUCKET, [present, missing], tmp_path)
        stub.assert_no_pending_responses()
    assert paths == {present: tmp_path / present, missing: tmp_path / missing}
    assert (tmp_path / missing).read_bytes() == b"jpeg"
    assert (tmp_path / present).read_bytes() == b"old"  # files are named by hash: never change
    assert not list(tmp_path.rglob("*.part"))


def test_sync_media_skips_a_missing_object(tmp_path):
    key = f"raw/durian/{3:064x}.jpg"
    s3 = fake_s3()
    with Stubber(s3) as stub:
        stub.add_client_error("get_object", "NoSuchKey", http_status_code=404)
        assert sync_media(s3, DATA_BUCKET, [key], tmp_path) == {}
    assert not (tmp_path / key).exists()


class _Body:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


# FiftyOne samples


def _record(**over) -> ImageRecord:
    base = {
        "image_id": 1,
        "s3_key": f"raw/durian/{1:064x}.jpg",
        "split": "train",
        "status": "labeled",
        "source": "seed",
        "created_at": datetime(2026, 10, 1, tzinfo=UTC),
        "label": "a",
        "label_source": "seed",
        "labeled_by": None,
        "label_round": None,
        "versions": [0, 1],
        "new_label": None,
        "new_label_round": None,
        "app_prediction": None,
        "app_confidence": None,
    }
    return ImageRecord(**{**base, **over})


def test_sample_fields(tmp_path):
    pytest.importorskip("fiftyone")
    from viewer.samples import to_sample

    r2 = uuid.uuid4()
    sample = to_sample(
        _record(
            label=None,
            label_source=None,
            versions=[],
            status="unlabeled",
            source="edge",
            new_label="b",
            new_label_round=r2,
            app_prediction="a",
            app_confidence=0.6,
        ),
        tmp_path / "x.jpg",
    )
    assert sample.filepath == str(tmp_path / "x.jpg")
    assert sample["image_id"] == 1
    assert sorted(sample.tags) == ["edge", "train", "unlabeled"]
    assert sample["ground_truth"] is None
    assert sample["new_label"].label == "b"
    assert sample["new_label_round"] == str(r2)
    assert (sample["app_prediction"].label, sample["app_prediction"].confidence) == ("a", 0.6)
    assert sample["versions"] == []
    assert sample["first_version"] is None


def test_labeled_sample(tmp_path):
    pytest.importorskip("fiftyone")
    from viewer.samples import to_sample

    sample = to_sample(_record(), tmp_path / "x.jpg")
    assert sample["ground_truth"].label == "a"
    assert sample["label_source"] == "seed"
    assert (sample["versions"], sample["first_version"]) == ([0, 1], 0)
    assert sample["new_label"] is None and sample["app_prediction"] is None


# One sync pass


def test_sync_once_publishes_every_dataset_with_its_local_images(db, durian, tmp_path):
    from viewer.main import sync_once

    dataset(db, "rocole", crop(db, "coffee", "Cà phê"), "Lá cà phê")  # no images yet
    published = {}
    missing = f"raw/durian/{6:064x}.jpg"

    def fake_media(s3, bucket, keys, root):
        assert (s3, bucket, root) == ("s3", DATA_BUCKET, tmp_path)
        return {k: root / k for k in keys if k != missing}

    counts = sync_once(
        db,
        "s3",
        DATA_BUCKET,
        tmp_path,
        publish=lambda name, pairs: published.update({name: pairs}),
        media=fake_media,
    )
    assert counts == {"durian": 6, "rocole": 0}
    assert set(published) == {"durian", "rocole"}
    pairs = published["durian"]
    assert [r.image_id for r, _ in pairs] == sorted(durian["ids"][n] for n in (1, 2, 3, 4, 5, 7))
    record, path = pairs[0]
    assert path == tmp_path / record.s3_key
