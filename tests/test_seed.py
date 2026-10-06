from collections import Counter
from pathlib import Path

import pytest

from cli.main import main
from common.dataset import DatasetConfig, SplitConfig
from db.seed import SeedError, assign_splits, scan, split_counts

ROOT = Path(__file__).resolve().parents[1]
JPEG = b"\xff\xd8\xff\xe0"


def make_folder(root: Path, per_label: dict[str, int], start: int = 0) -> Path:
    for label, n in per_label.items():
        (root / label).mkdir(parents=True, exist_ok=True)
        for i in range(start, start + n):
            (root / label / f"{i}.jpg").write_bytes(JPEG + f"{label}-{i}".encode())
    return root


def test_example_dataset_config_loads():
    cfg = DatasetConfig.from_yaml(ROOT / "configs/datasets/rocole.yaml")
    assert cfg.name == "rocole"
    assert cfg.split.test > 0
    assert cfg.codes == ["healthy", "red_spider_mite", "rust"]
    assert all(c.display_name for c in cfg.classes)


def test_class_codes_are_machine_names():
    with pytest.raises(ValueError):
        DatasetConfig(
            name="x",
            classes=[
                {"code": "Leaf Blight", "display_name": "A"},
                {"code": "b", "display_name": "B"},
            ],
            split={"val": 0.1, "test": 0.1},
        )


@pytest.mark.parametrize(
    "split", [{"val": 0.5, "test": 0.5}, {"val": 0.1, "test": 0.0}, {"val": -0.1, "test": 0.2}]
)
def test_split_config_rejects(split):
    with pytest.raises(ValueError):
        SplitConfig(**split)


def test_dataset_config_rejects_duplicate_codes():
    with pytest.raises(ValueError, match="unique"):
        DatasetConfig(
            name="x",
            classes=[{"code": "a", "display_name": "A"}, {"code": "a", "display_name": "B"}],
            split={"val": 0.1, "test": 0.1},
        )


def test_scan_reads_folders_and_dedupes(tmp_path):
    make_folder(tmp_path, {"a": 3, "b": 2})
    (tmp_path / "a" / "copy.jpg").write_bytes((tmp_path / "a" / "0.jpg").read_bytes())
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "a" / ".DS_Store").write_bytes(b"x")
    images = scan(tmp_path, ["a", "b"])
    assert Counter(i.label for i in images) == {"a": 3, "b": 2}


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda p: (p / "c").mkdir(), "not in the label list"),
        (lambda p: __import__("shutil").rmtree(p / "b"), "no folder for labels"),
        (lambda p: (p / "a" / "x.png").write_bytes(b"\x89PNG"), "not JPEG"),
        (lambda p: (p / "a" / "x.jpg").write_bytes(b"\x89PNG"), "not JPEG"),
        (
            lambda p: (p / "b" / "dup.jpg").write_bytes((p / "a" / "0.jpg").read_bytes()),
            "same image under",
        ),
    ],
)
def test_scan_rejects(tmp_path, setup, message):
    make_folder(tmp_path, {"a": 3, "b": 2})
    setup(tmp_path)
    with pytest.raises(SeedError, match=message):
        scan(tmp_path, ["a", "b"])


def test_scan_rejects_empty_label(tmp_path):
    make_folder(tmp_path, {"a": 3})
    (tmp_path / "b").mkdir()
    with pytest.raises(SeedError, match="no images for labels"):
        scan(tmp_path, ["a", "b"])


def test_split_is_stratified_and_deterministic(tmp_path):
    images = scan(make_folder(tmp_path, {"a": 100, "b": 20}), ["a", "b"])
    cfg = SplitConfig(val=0.15, test=0.15, seed=7)
    splits = assign_splits(images, cfg)
    counts = split_counts(images, splits)
    assert counts["test"] == {"a": 15, "b": 3}
    assert counts["val"] == {"a": 15, "b": 3}
    assert counts["train"] == {"a": 70, "b": 14}
    # File order does not matter; the seed does.
    assert assign_splits(list(reversed(images)), cfg) == splits
    assert assign_splits(images, SplitConfig(val=0.15, test=0.15, seed=8)) != splits


def test_seed_dry_run(tmp_path, capsys):
    make_folder(tmp_path / "imgs", {"healthy": 20, "red_spider_mite": 10, "rust": 10})
    assert (
        main(
            [
                "seed",
                str(ROOT / "configs/datasets/rocole.yaml"),
                str(tmp_path / "imgs"),
                "--dry-run",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "dry run: 40 images" in out
    assert "test       7" in out


def test_seed_needs_bucket(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DATA_BUCKET", raising=False)
    make_folder(tmp_path, {"healthy": 2, "red_spider_mite": 2, "rust": 2})
    assert main(["seed", str(ROOT / "configs/datasets/rocole.yaml"), str(tmp_path)]) == 1
    assert "--bucket" in capsys.readouterr().out


def test_s3_store_exists_and_put(tmp_path):
    import boto3
    from botocore.stub import Stubber

    from db.seed import S3Store

    client = boto3.client(
        "s3", region_name="ap-southeast-1", aws_access_key_id="x", aws_secret_access_key="x"
    )
    store = S3Store("bucket", client=client)
    with Stubber(client) as stub:
        stub.add_response("head_object", {}, {"Bucket": "bucket", "Key": "raw/a.jpg"})
        stub.add_client_error("head_object", "404", http_status_code=404)
        stub.add_client_error("head_object", "403", http_status_code=403)
        assert store.exists("raw/a.jpg") is True
        assert store.exists("raw/b.jpg") is False
        with pytest.raises(Exception, match="403"):
            store.exists("raw/c.jpg")
