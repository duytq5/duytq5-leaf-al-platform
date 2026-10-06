"""Seed a dataset: upload its images to raw/ and record them with a fixed split.

Input is a folder with one sub-folder per label (ImageFolder layout), each
holding JPEG files. Every image is named by its SHA-256, so re-running the seed
skips images that are already in S3 and in Postgres.

The train/val/test split is stratified per class and fixed: each image's place
depends only on its SHA-256 and the split seed, never on file order. Images
already in Postgres keep the split they were given the first time.

What lands where:
  train images  status 'unlabeled' (the AL pool); ground truth only in oracle_labels
  val, test     status 'labeled'; ground truth in labels (source 'seed', no round)
                and oracle_labels
"""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import psycopg

from common.dataset import DatasetConfig, SplitConfig
from common.s3keys import raw_key
from common.types import ImageSource, ImageStatus, Split

JPEG_SUFFIXES = {".jpg", ".jpeg"}
JPEG_MAGIC = b"\xff\xd8\xff"


class SeedError(ValueError):
    pass


@dataclass(frozen=True)
class SourceImage:
    sha256: str
    label: str
    path: Path


@dataclass
class SeedReport:
    dataset: str
    split_counts: dict[str, Counter] = field(default_factory=dict)  # split -> label -> n
    uploaded: int = 0
    already_in_s3: int = 0
    new_rows: int = 0
    existing_rows: int = 0
    kept_existing_split: int = 0


class ObjectStore(Protocol):
    def exists(self, key: str) -> bool: ...
    def put_jpeg(self, key: str, path: Path) -> None: ...


class S3Store:
    def __init__(self, bucket: str, client=None) -> None:
        import boto3

        self.bucket = bucket
        self.client = client or boto3.client("s3")

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def put_jpeg(self, key: str, path: Path) -> None:
        self.client.upload_file(
            str(path), self.bucket, key, ExtraArgs={"ContentType": "image/jpeg"}
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(source: Path, labels: list[str]) -> list[SourceImage]:
    """Read <source>/<label>/*.jpg. Fails on unknown or empty label folders,
    non-JPEG files, and the same image filed under two labels."""
    if not source.is_dir():
        raise SeedError(f"{source} is not a directory")
    folders = {p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")}
    if unknown := sorted(folders - set(labels)):
        raise SeedError(f"folders not in the label list: {unknown}")
    if missing := [y for y in labels if y not in folders]:
        raise SeedError(f"no folder for labels: {missing}")

    by_sha: dict[str, SourceImage] = {}
    bad: list[str] = []
    for label in labels:
        for path in sorted((source / label).rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in JPEG_SUFFIXES:
                bad.append(str(path))
                continue
            with open(path, "rb") as f:
                if f.read(3) != JPEG_MAGIC:
                    bad.append(str(path))
                    continue
            sha = _sha256(path)
            seen = by_sha.get(sha)
            if seen and seen.label != label:
                raise SeedError(
                    f"same image under {seen.label!r} and {label!r}: {seen.path}, {path}"
                )
            by_sha.setdefault(sha, SourceImage(sha, label, path))
    if bad:
        raise SeedError(f"{len(bad)} files are not JPEG, e.g. {bad[:3]}")
    if empty := [y for y in labels if not any(i.label == y for i in by_sha.values())]:
        raise SeedError(f"no images for labels: {empty}")
    return list(by_sha.values())


def assign_splits(images: Iterable[SourceImage], cfg: SplitConfig) -> dict[str, Split]:
    """Stratified, order-independent split: within each class, images are
    ranked by sha256(seed:image_sha) and the first round(n*test) go to test,
    the next round(n*val) to val, the rest to train."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for img in images:
        by_label[img.label].append(img.sha256)

    out: dict[str, Split] = {}
    for shas in by_label.values():
        ranked = sorted(shas, key=lambda s: hashlib.sha256(f"{cfg.seed}:{s}".encode()).hexdigest())
        n_test = round(len(ranked) * cfg.test)
        n_val = round(len(ranked) * cfg.val)
        for i, sha in enumerate(ranked):
            out[sha] = (
                Split.TEST if i < n_test else Split.VAL if i < n_test + n_val else Split.TRAIN
            )
    return out


def _upload(images: list[SourceImage], dataset: str, store: ObjectStore, report: SeedReport):
    def one(img: SourceImage) -> bool:
        key = raw_key(dataset, img.sha256)
        if store.exists(key):
            return False
        store.put_jpeg(key, img.path)
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, images))
    report.uploaded = sum(results)
    report.already_in_s3 = len(results) - report.uploaded


def _record(conn: psycopg.Connection, cfg: DatasetConfig, images, splits, report: SeedReport):
    with conn.transaction():
        conn.execute(
            "INSERT INTO datasets (name, labels) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
            (cfg.name, cfg.labels),
        )
        dataset_id, stored = conn.execute(
            "SELECT id, labels FROM datasets WHERE name = %s", (cfg.name,)
        ).fetchone()
        if stored != cfg.labels:
            raise SeedError(
                f"label list for {cfg.name!r} is {stored} in the database, {cfg.labels} in the "
                "config; the order is the model's output order and cannot change"
            )

        before = conn.execute(
            "SELECT count(*) FROM images WHERE dataset_id = %s", (dataset_id,)
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO images (dataset_id, sha256, s3_key, status, split, source)"
                " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (dataset_id, sha256) DO NOTHING",
                [
                    (
                        dataset_id,
                        img.sha256,
                        raw_key(cfg.name, img.sha256),
                        ImageStatus.UNLABELED
                        if splits[img.sha256] is Split.TRAIN
                        else ImageStatus.LABELED,
                        splits[img.sha256],
                        ImageSource.SEED,
                    )
                    for img in images
                ],
            )
        rows = {
            sha: (image_id, Split(split))
            for sha, image_id, split in conn.execute(
                "SELECT sha256, id, split FROM images WHERE dataset_id = %s", (dataset_id,)
            )
        }
        report.new_rows = len(rows) - before
        report.existing_rows = len(images) - report.new_rows

        truth = {
            image_id: label
            for image_id, label in conn.execute(
                "SELECT o.image_id, o.label FROM oracle_labels o"
                " JOIN images i ON i.id = o.image_id WHERE i.dataset_id = %s",
                (dataset_id,),
            )
        }
        conflicts = [
            img.path for img in images if truth.get(rows[img.sha256][0], img.label) != img.label
        ]
        if conflicts:
            raise SeedError(
                f"{len(conflicts)} images have a different label than when first "
                f"seeded, e.g. {conflicts[:3]}"
            )

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO oracle_labels (image_id, label) VALUES (%s, %s)"
                " ON CONFLICT (image_id) DO NOTHING",
                [(rows[img.sha256][0], img.label) for img in images],
            )
            cur.executemany(
                "INSERT INTO labels (image_id, label, round_id, source)"
                " VALUES (%s, %s, NULL, 'seed')"
                " ON CONFLICT (image_id, round_id) DO NOTHING",
                [
                    (rows[img.sha256][0], img.label)
                    for img in images
                    if rows[img.sha256][1] is not Split.TRAIN
                ],
            )

    final = {sha: split for sha, (_, split) in rows.items()}
    report.kept_existing_split = sum(1 for img in images if final[img.sha256] != splits[img.sha256])
    report.split_counts = split_counts(images, final)


def plan(cfg: DatasetConfig, source: Path) -> tuple[list[SourceImage], dict[str, Split]]:
    images = scan(source, cfg.labels)
    return images, assign_splits(images, cfg.split)


def seed_dataset(
    conn: psycopg.Connection, cfg: DatasetConfig, source: Path, store: ObjectStore
) -> SeedReport:
    """Upload first, then record, so a row never points at a missing object.
    Safe to re-run after a failure at any point."""
    images, splits = plan(cfg, source)
    report = SeedReport(dataset=cfg.name)
    _upload(images, cfg.name, store, report)
    _record(conn, cfg, images, splits, report)
    return report


def split_counts(images: list[SourceImage], splits: dict[str, Split]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for img in images:
        out[splits[img.sha256].value][img.label] += 1
    return dict(out)
