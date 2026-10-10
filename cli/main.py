import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from common.config import TrainConfig
from common.dataset import CropsConfig, DatasetConfig
from common.round import RoundConfig
from selection import describe_strategies, from_config


def cmd_strategies(args: argparse.Namespace) -> int:
    print(json.dumps(describe_strategies(), indent=2))
    return 0


def cmd_check_round(args: argparse.Namespace) -> int:
    """Validate a round config, its strategy params and its train config."""
    try:
        cfg = RoundConfig.from_yaml(args.config)
        strategy = from_config(cfg.selection)
        train = TrainConfig.from_yaml(cfg.train_config)  # relative to the repo root
    except KeyError as e:
        print(e.args[0])
        return 1
    except (OSError, ValidationError) as e:
        print(e)
        return 1
    if train.data.dataset != cfg.dataset:
        print(f"train config dataset {train.data.dataset!r} != round dataset {cfg.dataset!r}")
        return 1
    print(
        f"ok: {cfg.mode} round on {cfg.dataset}, strategy={cfg.selection.strategy} "
        f"params={strategy.params.model_dump()} k={cfg.selection.k} "
        f"needs={sorted(strategy.requires) or 'nothing'}"
    )
    return 0


def cmd_db_migrate(args: argparse.Namespace) -> int:
    from db import connect
    from db.migrate import migrate

    with connect() as conn:
        applied = migrate(conn)
    print(f"applied: {', '.join(applied)}" if applied else "schema is up to date")
    return 0


def _print_split(counts: dict) -> None:
    for split in ("train", "val", "test"):
        c = counts.get(split, {})
        detail = ", ".join(f"{label}={n}" for label, n in sorted(c.items()))
        print(f"  {split:5} {sum(c.values()):6}  {detail}")


def cmd_seed(args: argparse.Namespace) -> int:
    """Upload a dataset to raw/ and record it with a fixed split (retry-safe)."""
    from db.seed import SeedError, check_crop, plan, split_counts

    try:
        cfg = DatasetConfig.from_yaml(args.config)
        crops = CropsConfig.from_yaml(args.crops)
        check_crop(cfg, crops)
        if args.dry_run:
            images, splits = plan(cfg, Path(args.source))
            print(f"dry run: {len(images)} images for {cfg.name}, classes {cfg.codes}")
            _print_split(split_counts(images, splits))
            return 0
        bucket = args.bucket or os.environ.get("DATA_BUCKET")
        if not bucket:
            print("set --bucket or DATA_BUCKET")
            return 1

        from db import connect
        from db.seed import S3Store, seed_dataset

        with connect() as conn:
            report = seed_dataset(conn, cfg, Path(args.source), S3Store(bucket), crops)
    except (OSError, ValidationError, SeedError) as e:
        print(e)
        return 1
    print(
        f"{cfg.name}: uploaded {report.uploaded}, already in S3 {report.already_in_s3}; "
        f"new rows {report.new_rows}, existing {report.existing_rows}; "
        f"v0 labels added {report.v0_added}"
        + (
            f", kept earlier split for {report.kept_existing_split}"
            if report.kept_existing_split
            else ""
        )
    )
    _print_split(report.split_counts)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="al", description="Active-learning operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("strategies", help="list strategies, required inputs and parameters")
    check = sub.add_parser("check-round", help="validate a round config YAML")
    check.add_argument("config")

    db = sub.add_parser("db", help="database commands (needs DATABASE_URL)")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("migrate", help="create or update the tables; safe to re-run")

    seed = sub.add_parser("seed", help="upload a dataset to S3 and record it with a fixed split")
    seed.add_argument("config", help="configs/datasets/<name>.yaml")
    seed.add_argument("source", help="folder with one sub-folder of JPEGs per label")
    seed.add_argument("--bucket", help="data bucket (default: $DATA_BUCKET)")
    seed.add_argument(
        "--crops", default="configs/crops.yaml", help="crop list (default: configs/crops.yaml)"
    )
    seed.add_argument(
        "--dry-run", action="store_true", help="show the split; no upload, no database"
    )

    args = parser.parse_args(argv)
    if args.command == "db":
        return {"migrate": cmd_db_migrate}[args.db_command](args)
    handlers = {"strategies": cmd_strategies, "check-round": cmd_check_round, "seed": cmd_seed}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
