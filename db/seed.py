"""Seed a dataset: upload its images to raw/ and record them with a fixed split.

Input is a folder with one sub-folder per label (ImageFolder layout), each
holding JPEG files. Every image is named by its SHA-256, so re-running the seed
skips images that are already in S3 and in Postgres.

The train/val/test split is stratified per class and fixed: each image's place
depends only on its SHA-256 (or its group, see SplitConfig.group_pattern) and
the split seed, never on file order. Images of one group, such as the views of
one plant, always share a split. Images already in Postgres keep the split they
were given the first time.

What lands where:
  train images  status 'unlabeled' (the AL pool); ground truth only in oracle_labels
  initial set   a share of each class's new train images (split.initial_labeled),
                until a round has used v0: status 'labeled', label in labels
                (source 'seed', no round)
  val, test     status 'labeled'; ground truth in labels (source 'seed', no round)
  every image   ground truth in oracle_labels

Crops from configs/crops.yaml are upserted by code (an existing crop only gets
its names updated), and the dataset must name one of them.

The seed also makes dataset version v0: the seed labels (val, test and the
initial train set), which the first round trains on. Once a round has used v0
it is frozen, and adding seed labels is refused (they would change v0 under
that round).
"""

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import psycopg

from common.dataset import CropsConfig, DatasetConfig, SplitConfig
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
    initial_labeled: int = 0  # train images labeled at seeding in this run
    conflicting_dropped: int = 0  # images left out: filed under two labels
    v0_added: int = 0


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


def scan(source: Path, labels: list[str], conflicting: set[str] | None = None) -> list[SourceImage]:
    """Read <source>/<class code>/*.jpg. Fails on unknown or empty label folders
    and non-JPEG files.

    Byte-identical copies are one image (the first path in class order is
    kept). An image filed under two different labels has no trustworthy label,
    so every copy of it is left out; its SHA-256 goes into `conflicting`.
    RoCoLe has 12 such pairs, which the research code drops the same way."""
    if not source.is_dir():
        raise SeedError(f"{source} is not a directory")
    folders = {p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")}
    if unknown := sorted(folders - set(labels)):
        raise SeedError(f"folders not in the label list: {unknown}")
    if missing := [y for y in labels if y not in folders]:
        raise SeedError(f"no folder for labels: {missing}")

    by_sha: dict[str, SourceImage] = {}
    dropped: set[str] = set()
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
                dropped.add(sha)
            by_sha.setdefault(sha, SourceImage(sha, label, path))
    for sha in dropped:
        del by_sha[sha]
    if conflicting is not None:
        conflicting.update(dropped)
    if bad:
        raise SeedError(f"{len(bad)} files are not JPEG, e.g. {bad[:3]}")
    if empty := [y for y in labels if not any(i.label == y for i in by_sha.values())]:
        raise SeedError(f"no images for labels: {empty}")
    return list(by_sha.values())


def group_key(img: SourceImage, cfg: SplitConfig) -> str:
    """The unit the split keeps together: the group_pattern match on the file
    name, or the image itself."""
    if cfg.group_pattern is None:
        return img.sha256
    m = re.search(cfg.group_pattern, img.path.name)
    if not m:
        raise SeedError(f"{img.path.name!r} does not match group_pattern {cfg.group_pattern!r}")
    return m.group(1) if m.groups() else m.group(0)


def assign_splits(images: Iterable[SourceImage], cfg: SplitConfig) -> dict[str, Split]:
    """Stratified, grouped, order-independent split. Each group counts under
    its most common class. Within a class, groups are ranked by
    sha256(seed:group) and fill test up to round(n*test) images, then val up
    to round(n*val), then the rest go to train (n = images of that class)."""
    groups: dict[str, list[SourceImage]] = defaultdict(list)
    for img in images:
        groups[group_key(img, cfg)].append(img)

    by_label: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        counts = Counter(i.label for i in members)
        label = min(counts, key=lambda y: (-counts[y], y))
        by_label[label].append(key)

    out: dict[str, Split] = {}
    for keys in by_label.values():
        ranked = sorted(keys, key=lambda k: hashlib.sha256(f"{cfg.seed}:{k}".encode()).hexdigest())
        n = sum(len(groups[k]) for k in ranked)
        n_test, n_val = round(n * cfg.test), round(n * cfg.val)
        taken = {Split.TEST: 0, Split.VAL: 0}
        for key in ranked:
            if taken[Split.TEST] < n_test:
                split = Split.TEST
            elif taken[Split.VAL] < n_val:
                split = Split.VAL
            else:
                split = Split.TRAIN
            if split in taken:
                taken[split] += len(groups[key])
            for img in groups[key]:
                out[img.sha256] = split
    return out


def pick_initial(
    images: Iterable[SourceImage], splits: dict[str, Split], cfg: SplitConfig
) -> set[str]:
    """The train images labeled at seeding: round(n*initial_labeled) of each
    class's n train images, ranked by sha256(seed:initial:image_sha)."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for img in images:
        if splits[img.sha256] is Split.TRAIN:
            by_label[img.label].append(img.sha256)
    out: set[str] = set()
    for shas in by_label.values():
        ranked = sorted(
            shas, key=lambda s: hashlib.sha256(f"{cfg.seed}:initial:{s}".encode()).hexdigest()
        )
        out.update(ranked[: round(len(ranked) * cfg.initial_labeled)])
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


def _record(
    conn: psycopg.Connection,
    cfg: DatasetConfig,
    crops: CropsConfig,
    images,
    splits,
    initial: set[str],
    report: SeedReport,
):
    with conn.transaction():
        _upsert_crops(conn, crops)
        dataset_id = _dataset(conn, cfg)
        class_ids = dict(
            conn.execute("SELECT code, id FROM classes WHERE dataset_id = %s", (dataset_id,))
        )

        # Only images new to the database can join the initial labeled set: an
        # existing train image keeps its state, whatever later rounds did to it.
        existing = {
            sha
            for (sha,) in conn.execute(
                "SELECT sha256 FROM images WHERE dataset_id = %s", (dataset_id,)
            )
        }
        initial = {sha for sha in initial if sha not in existing}
        # Once a round trained on v0, v0 is frozen: new train images join the pool.
        if conn.execute(
            "SELECT 1 FROM rounds r JOIN dataset_versions v ON v.id = r.base_version_id"
            " WHERE v.dataset_id = %s AND v.version = 0",
            (dataset_id,),
        ).fetchone():
            initial = set()
        before = len(existing)
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
                        if splits[img.sha256] is Split.TRAIN and img.sha256 not in initial
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

        truth = dict(
            conn.execute(
                "SELECT o.image_id, o.class_id FROM oracle_labels o"
                " JOIN images i ON i.id = o.image_id WHERE i.dataset_id = %s",
                (dataset_id,),
            )
        )
        conflicts = [
            img.path
            for img in images
            if truth.get(rows[img.sha256][0], class_ids[img.label]) != class_ids[img.label]
        ]
        if conflicts:
            raise SeedError(
                f"{len(conflicts)} images have a different label than when first "
                f"seeded, e.g. {conflicts[:3]}"
            )

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO oracle_labels (image_id, class_id) VALUES (%s, %s)"
                " ON CONFLICT (image_id) DO NOTHING",
                [(rows[img.sha256][0], class_ids[img.label]) for img in images],
            )
            cur.executemany(
                "INSERT INTO labels (image_id, class_id, round_id, source)"
                " VALUES (%s, %s, NULL, 'seed')"
                " ON CONFLICT (image_id, round_id) DO NOTHING",
                [
                    (rows[img.sha256][0], class_ids[img.label])
                    for img in images
                    if rows[img.sha256][1] is not Split.TRAIN or img.sha256 in initial
                ],
            )
        report.initial_labeled = len(initial)
        report.v0_added = _fill_v0(conn, cfg.name, dataset_id)

    final = {sha: split for sha, (_, split) in rows.items()}
    report.kept_existing_split = sum(1 for img in images if final[img.sha256] != splits[img.sha256])
    report.split_counts = split_counts(images, final)


def _upsert_crops(conn: psycopg.Connection, crops: CropsConfig) -> None:
    """Add new crops; an existing crop only gets its names updated."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO crops (code, display_name, description) VALUES (%s, %s, %s)"
            " ON CONFLICT (code) DO UPDATE"
            " SET display_name = EXCLUDED.display_name, description = EXCLUDED.description",
            [(c.code, c.display_name, c.description) for c in crops.crops],
        )


def _dataset(conn: psycopg.Connection, cfg: DatasetConfig) -> int:
    """Create the dataset and its classes if needed; update its readable names.
    The crop and the class codes and their order cannot change, because the
    model's output order depends on them."""
    (dataset_id, crop) = conn.execute(
        "INSERT INTO datasets (name, crop_id, display_name, description)"
        " SELECT %s, id, %s, %s FROM crops WHERE code = %s"
        " ON CONFLICT (name) DO UPDATE"
        " SET display_name = EXCLUDED.display_name, description = EXCLUDED.description"
        " RETURNING id, (SELECT code FROM crops WHERE id = datasets.crop_id)",
        (cfg.name, cfg.display_name, cfg.description, cfg.crop),
    ).fetchone()
    if crop != cfg.crop:
        raise SeedError(
            f"{cfg.name!r} belongs to crop {crop!r} in the database, {cfg.crop!r} in the config"
        )
    stored = [
        code
        for (code,) in conn.execute(
            "SELECT code FROM classes WHERE dataset_id = %s ORDER BY position", (dataset_id,)
        )
    ]
    if not stored:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO classes (dataset_id, code, position, display_name, description)"
                " VALUES (%s, %s, %s, %s, %s)",
                [
                    (dataset_id, c.code, i, c.display_name, c.description)
                    for i, c in enumerate(cfg.classes)
                ],
            )
    elif stored != cfg.codes:
        raise SeedError(
            f"classes of {cfg.name!r} are {stored} in the database, {cfg.codes} in the "
            "config; the order is the model's output order and cannot change"
        )
    return dataset_id


def _fill_v0(conn: psycopg.Connection, dataset: str, dataset_id: int) -> int:
    """Create v0 if needed and add the seed labels it is missing."""
    conn.execute(
        "INSERT INTO dataset_versions (dataset_id, version) VALUES (%s, 0)"
        " ON CONFLICT (dataset_id, version) DO NOTHING",
        (dataset_id,),
    )
    (v0,) = conn.execute(
        "SELECT id FROM dataset_versions WHERE dataset_id = %s AND version = 0", (dataset_id,)
    ).fetchone()
    missing = conn.execute(
        "SELECT l.id, l.image_id FROM labels l JOIN images i ON i.id = l.image_id"
        " WHERE i.dataset_id = %s AND l.source = 'seed' AND NOT EXISTS ("
        "   SELECT 1 FROM dataset_version_labels v"
        "   WHERE v.version_id = %s AND v.image_id = l.image_id)",
        (dataset_id, v0),
    ).fetchall()
    if not missing:
        return 0
    if conn.execute("SELECT 1 FROM rounds WHERE base_version_id = %s", (v0,)).fetchone():
        raise SeedError(
            f"{len(missing)} new seed labels for {dataset!r}, but a round already "
            "trained on v0, which must not change; seed them into a new dataset"
        )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO dataset_version_labels (version_id, image_id, label_id)"
            " VALUES (%s, %s, %s)",
            [(v0, image_id, label_id) for label_id, image_id in missing],
        )
    return len(missing)


def plan(
    cfg: DatasetConfig, source: Path
) -> tuple[list[SourceImage], dict[str, Split], set[str], int]:
    """Images, their split, the train images labeled at seeding, and how many
    images were left out for being filed under two labels."""
    conflicting: set[str] = set()
    images = scan(source, cfg.codes, conflicting)
    splits = assign_splits(images, cfg.split)
    return images, splits, pick_initial(images, splits, cfg.split), len(conflicting)


def check_crop(cfg: DatasetConfig, crops: CropsConfig) -> None:
    if cfg.crop not in crops.codes:
        raise SeedError(
            f"unknown crop {cfg.crop!r} for {cfg.name!r}; crops are {sorted(crops.codes)}"
        )


def seed_dataset(
    conn: psycopg.Connection,
    cfg: DatasetConfig,
    source: Path,
    store: ObjectStore,
    crops: CropsConfig,
) -> SeedReport:
    """Upload first, then record, so a row never points at a missing object.
    Safe to re-run after a failure at any point. Crops are upserted from
    configs/crops.yaml; the dataset's crop must be one of them."""
    check_crop(cfg, crops)
    images, splits, initial, conflicting = plan(cfg, source)
    report = SeedReport(dataset=cfg.name, conflicting_dropped=conflicting)
    _upload(images, cfg.name, store, report)
    _record(conn, cfg, crops, images, splits, initial, report)
    return report


def split_counts(images: list[SourceImage], splits: dict[str, Split]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for img in images:
        out[splits[img.sha256].value][img.label] += 1
    return dict(out)
