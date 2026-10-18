"""Repair of app images stored before the upload fix (db/rehash.py): their
raw/ file is the clean JPEG, but the row and the key named the upload's
sha256, so the worker refused them ("does not match its sha256").

Against a real Postgres (TEST_DATABASE_URL); S3 is an in-memory fake.
"""

import hashlib

import pytest

from db.rehash import apply, plan
from tests.apidb import crop, dataset


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def put_jpeg(self, key: str, body: bytes) -> None:
        self.objects[key] = body


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def world(db):
    return {"rocole": dataset(db, "rocole", crop(db, "coffee", "Cà phê"), "Lá cà phê")}


def _image(db, world, sha: str, source: str = "edge") -> int:
    return db.execute(
        "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
        " VALUES (%s, %s, %s, 'unlabeled', 'train', %s) RETURNING id",
        (world["rocole"], sha, f"raw/rocole/{sha}.jpg", source),
    ).fetchone()[0]


def _row(db, image_id: int) -> tuple:
    return db.execute("SELECT sha256, s3_key FROM images WHERE id = %s", (image_id,)).fetchone()


def test_an_edge_image_under_the_upload_hash_is_renamed(db, world):
    store = FakeStore()
    upload_sha = _sha(b"what the phone sent")
    clean = b"the clean jpeg"
    store.objects[f"raw/rocole/{upload_sha}.jpg"] = clean
    image_id = _image(db, world, upload_sha)

    fixes = plan(db, store)
    assert [(f.image_id, f.old_sha, f.new_sha) for f in fixes] == [
        (image_id, upload_sha, _sha(clean))
    ]
    assert _row(db, image_id) == (upload_sha, f"raw/rocole/{upload_sha}.jpg")  # plan only

    apply(db, store, fixes)
    new_key = f"raw/rocole/{_sha(clean)}.jpg"
    assert _row(db, image_id) == (_sha(clean), new_key)
    assert store.objects[new_key] == clean
    # The old file stays: earlier Label Studio tasks may still open it.
    assert store.objects[f"raw/rocole/{upload_sha}.jpg"] == clean


def test_images_whose_file_matches_are_left_alone(db, world):
    store = FakeStore()
    for body, source in ((b"seed file", "seed"), (b"fixed upload", "edge")):
        store.objects[f"raw/rocole/{_sha(body)}.jpg"] = body
        _image(db, world, _sha(body), source)
    assert plan(db, store) == []


def test_a_seed_image_is_never_touched(db, world):
    """Only app uploads were re-encoded; a seed file that does not match is
    a different problem and stays for a person to look at."""
    store = FakeStore()
    sha = _sha(b"seed")
    store.objects[f"raw/rocole/{sha}.jpg"] = b"something else"
    _image(db, world, sha, "seed")
    assert plan(db, store) == []


def test_a_missing_file_is_reported_not_fixed(db, world):
    image_id = _image(db, world, _sha(b"gone"))
    with pytest.raises(LookupError, match=f"image {image_id}"):
        plan(db, FakeStore())


def test_applying_twice_changes_nothing(db, world):
    store = FakeStore()
    upload_sha = _sha(b"phone")
    store.objects[f"raw/rocole/{upload_sha}.jpg"] = b"clean"
    image_id = _image(db, world, upload_sha)
    apply(db, store, plan(db, store))
    after = _row(db, image_id)
    assert plan(db, store) == []
    assert _row(db, image_id) == after


# al images rehash


def _broken(db, world) -> tuple[FakeStore, int]:
    store = FakeStore()
    upload_sha = _sha(b"phone")
    store.objects[f"raw/rocole/{upload_sha}.jpg"] = b"clean"
    return store, _image(db, world, upload_sha)


def _cli(monkeypatch, pg_url, store, *args) -> int:
    from cli.main import main
    from db import rehash

    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setattr(rehash, "S3Store", lambda bucket: store)
    return main(["images", "rehash", "--bucket", "data", *args])


def test_cli_shows_the_fixes_and_changes_nothing(db, world, pg_url, monkeypatch, capsys):
    store, image_id = _broken(db, world)
    before = _row(db, image_id)
    assert _cli(monkeypatch, pg_url, store) == 0
    out = capsys.readouterr().out
    assert f"image {image_id} (rocole)" in out and _sha(b"clean")[:12] in out
    assert "dry run" in out and "--apply" in out
    assert _row(db, image_id) == before


def test_cli_apply_fixes_the_rows(db, world, pg_url, monkeypatch, capsys):
    store, image_id = _broken(db, world)
    assert _cli(monkeypatch, pg_url, store, "--apply") == 0
    assert _row(db, image_id)[0] == _sha(b"clean")
    assert "fixed 1" in capsys.readouterr().out
    assert _cli(monkeypatch, pg_url, store) == 0
    assert "nothing to fix" in capsys.readouterr().out
