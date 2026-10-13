"""Upload validation (design doc F1 step 6): an app image lands in
incoming/{sha256}.jpg, S3 notifies a queue, and lambdas/upload.py checks it,
moves it to raw/{dataset}/{sha256}.jpg as a clean JPEG, adds it to the
dataset's train pool and links the waiting captures.

Against a real Postgres (TEST_DATABASE_URL); S3 is an in-memory fake.
"""

import hashlib
import io
import json
import uuid

import pytest

pytest.importorskip("PIL")
from PIL import Image

from lambdas import upload
from tests.apidb import crop, dataset, model


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def put_jpeg(self, key: str, body: bytes) -> None:
        self.objects[key] = body

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)


def _jpeg(size=(40, 20), color=(200, 30, 30), exif: Image.Exif | None = None) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, "JPEG", exif=exif or Image.Exif())
    return out.getvalue()


def _png(size=(30, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGBA", size, (0, 120, 0, 128)).save(out, "PNG")
    return out.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def world(db):
    durian = dataset(db, "durian", crop(db, "durian", "Sầu riêng"), "Lá sầu riêng")
    fruit = dataset(db, "durian_fruit", crop(db, "fruit", "Quả"), "Quả sầu riêng")
    return {
        "durian": durian,
        "fruit": fruit,
        "models": {durian: model(db, durian, "durian", 1), fruit: model(db, fruit, "fruit", 1)},
        "store": FakeStore(),
    }


def _capture(db, world, data: bytes, dataset_id: int | None = None, content_type="image/jpeg"):
    dataset_id = dataset_id or world["durian"]
    return db.execute(
        "INSERT INTO captures (capture_id, dataset_id, sha256, user_sub, model_version_id,"
        " top1_class_id, confidence, probs, captured_at, content_type, size_bytes)"
        " SELECT %s, %s, %s, 'sub-1', %s, c.id, 0.9, '{\"a\": 0.9, \"b\": 0.1}', now(), %s, %s"
        " FROM classes c WHERE c.dataset_id = %s AND c.code = 'a' RETURNING id",
        (
            uuid.uuid4(),
            dataset_id,
            _sha(data),
            world["models"][dataset_id],
            content_type,
            len(data),
            dataset_id,
        ),
    ).fetchone()[0]


def _upload(world, data: bytes) -> str:
    key = f"incoming/{_sha(data)}.jpg"
    world["store"].objects[key] = data
    return key


def _linked(db) -> list:
    return [r[0] for r in db.execute("SELECT image_id FROM captures ORDER BY id")]


def _images(db) -> list[tuple]:
    return db.execute(
        "SELECT d.name, i.sha256, i.s3_key, i.split, i.status, i.source"
        " FROM images i JOIN datasets d ON d.id = i.dataset_id ORDER BY i.id"
    ).fetchall()


# Accepted


def test_a_jpeg_joins_the_train_pool(db, world):
    data = _jpeg()
    _capture(db, world, data)
    key = _upload(world, data)
    assert upload.process(db, world["store"], key) == "stored"

    sha = _sha(data)
    raw = f"raw/durian/{sha}.jpg"
    assert _images(db) == [("durian", sha, raw, "train", "unlabeled", "edge")]
    assert _linked(db) == [db.execute("SELECT id FROM images").fetchone()[0]]
    store = world["store"].objects
    assert key not in store  # moved, not copied
    assert Image.open(io.BytesIO(store[raw])).format == "JPEG"


def test_exif_is_applied_then_removed(db, world):
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° clockwise to display
    exif[0x8825] = {2: (10.0, 45.0, 0.0)}  # GPS latitude: never kept
    data = _jpeg(size=(40, 20), exif=exif)
    _capture(db, world, data)
    upload.process(db, world["store"], _upload(world, data))

    out = Image.open(io.BytesIO(world["store"].objects[f"raw/durian/{_sha(data)}.jpg"]))
    assert out.size == (20, 40)  # stored the way the phone showed it
    assert len(out.getexif()) == 0
    assert "exif" not in out.info


def test_a_png_is_stored_as_jpeg(db, world):
    data = _png()
    _capture(db, world, data, content_type="image/png")
    assert upload.process(db, world["store"], _upload(world, data)) == "stored"
    out = Image.open(io.BytesIO(world["store"].objects[f"raw/durian/{_sha(data)}.jpg"]))
    assert (out.format, out.mode) == ("JPEG", "RGB")


def test_one_image_sent_for_two_datasets_joins_both(db, world):
    data = _jpeg()
    _capture(db, world, data)
    _capture(db, world, data, world["fruit"])
    _capture(db, world, data)  # a second capture of the same image
    upload.process(db, world["store"], _upload(world, data))
    assert [r[0] for r in _images(db)] == ["durian", "durian_fruit"]
    assert None not in _linked(db)
    assert len(set(_linked(db))) == 2
    assert f"raw/durian_fruit/{_sha(data)}.jpg" in world["store"].objects


def test_an_image_the_dataset_has_keeps_its_row_and_file(db, world):
    """A seed image keeps its split, status and original file; the capture is linked."""
    data = _jpeg()
    sha = _sha(data)
    raw = f"raw/durian/{sha}.jpg"
    world["store"].objects[raw] = b"original seed bytes"
    seed_id = db.execute(
        "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
        " VALUES (%s, %s, %s, 'labeled', 'test', 'seed') RETURNING id",
        (world["durian"], sha, raw),
    ).fetchone()[0]
    _capture(db, world, data)
    upload.process(db, world["store"], _upload(world, data))
    assert _images(db) == [("durian", sha, raw, "test", "labeled", "seed")]
    assert _linked(db) == [seed_id]
    assert world["store"].objects[raw] == b"original seed bytes"


def test_processing_again_changes_nothing(db, world):
    data = _jpeg()
    _capture(db, world, data)
    key = _upload(world, data)
    upload.process(db, world["store"], key)
    before = (_images(db), _linked(db))
    # The queue delivers twice, or the Lambda died before deleting incoming/.
    assert upload.process(db, world["store"], key) == "gone"
    world["store"].objects[key] = data
    upload.process(db, world["store"], key)
    assert (_images(db), _linked(db)) == before
    assert key not in world["store"].objects


# Refused: the file is deleted, nothing joins the pool, captures stay unlinked


def _refused(db, world, key: str) -> None:
    assert upload.process(db, world["store"], key) == "rejected"
    assert world["store"].objects == {}
    assert _images(db) == []
    assert set(_linked(db)) <= {None}


def test_not_an_image(db, world):
    data = b"MZ\x90\x00 definitely not a picture"
    _capture(db, world, data)
    _refused(db, world, _upload(world, data))


def test_a_truncated_jpeg(db, world):
    data = _jpeg(size=(200, 200))[:300]
    _capture(db, world, data)
    _refused(db, world, _upload(world, data))


def test_bytes_that_do_not_match_the_key(db, world):
    data = _jpeg()
    _capture(db, world, data)
    key = _upload(world, data)
    world["store"].objects[key] = _jpeg(color=(0, 0, 255))
    _refused(db, world, key)


def test_a_png_declared_as_jpeg(db, world):
    data = _png()
    _capture(db, world, data, content_type="image/jpeg")
    _refused(db, world, _upload(world, data))


def test_too_many_pixels(db, world, monkeypatch):
    monkeypatch.setattr(upload, "MAX_PIXELS", 500)
    data = _jpeg(size=(40, 20))
    _capture(db, world, data)
    _refused(db, world, _upload(world, data))


def test_an_upload_without_a_capture_is_deleted(db, world):
    _refused(db, world, _upload(world, _jpeg()))


def test_a_key_outside_the_layout_is_left_alone(db, world):
    world["store"].objects["incoming/notes.txt"] = b"x"
    assert upload.process(db, world["store"], "incoming/notes.txt") == "ignored"
    assert "incoming/notes.txt" in world["store"].objects


# The SQS handler


def _sqs(*bodies: dict) -> dict:
    return {
        "Records": [
            {"messageId": f"m{i}", "body": json.dumps(body)} for i, body in enumerate(bodies)
        ]
    }


def _s3_body(key: str) -> dict:
    return {"Records": [{"eventName": "ObjectCreated:Put", "s3": {"object": {"key": key}}}]}


def test_handler_processes_s3_records_and_reports_failures(db, world, monkeypatch):
    data = _jpeg()
    _capture(db, world, data)
    key = _upload(world, data)
    seen = []

    def process(conn, store, k):
        seen.append(k)
        if k == "incoming/boom.jpg":
            raise RuntimeError("postgres went away")
        return "stored"

    monkeypatch.setattr(upload, "process", process)
    monkeypatch.setattr(upload, "open_db", lambda: db)
    monkeypatch.setattr(upload, "_conn", None)
    monkeypatch.setattr(upload, "_store", lambda: world["store"])
    event = _sqs(
        {"Service": "Amazon S3", "Event": "s3:TestEvent"},  # sent once when S3 is wired up
        _s3_body(key.replace("incoming/", "incoming%2F")),
        _s3_body("incoming/boom.jpg"),
    )
    result = upload.handler(event, None)
    # The failed message comes back later from the queue (the host may be stopped).
    assert result == {"batchItemFailures": [{"itemIdentifier": "m2"}]}
    assert seen == [key, "incoming/boom.jpg"]
