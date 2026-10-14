import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from common.dataset import CropsConfig, DatasetConfig
from common.release import RELEASES_DIR, ReleaseExecution, ReleaseFile
from common.round import RoundConfig
from recipes import describe_recipes, get_recipe
from selection import describe_strategies, from_config


def cmd_strategies(args: argparse.Namespace) -> int:
    print(json.dumps(describe_strategies(), indent=2))
    return 0


def cmd_recipes(args: argparse.Namespace) -> int:
    print(json.dumps(describe_recipes(), indent=2))
    return 0


def cmd_check_round(args: argparse.Namespace) -> int:
    """Validate a round config: its dataset, recipe params and strategy params
    (no AWS, no torch; from the ops repo, --datasets points at the platform
    checkout's dataset configs)."""
    try:
        cfg = RoundConfig.from_yaml(args.config)
        recipe = get_recipe(cfg.train.recipe, cfg.train.params)
        strategy = from_config(cfg.selection)
    except KeyError as e:
        print(e.args[0])
        return 1
    except (OSError, ValidationError) as e:
        print(e)
        return 1
    dataset = Path(args.datasets) / f"{cfg.dataset}.yaml"
    if not dataset.is_file():
        print(f"no dataset {cfg.dataset} ({dataset} does not exist)")
        return 1
    version = "latest version" if cfg.base_version is None else f"v{cfg.base_version}"
    print(
        f"ok: round on {cfg.dataset} {version}"
        f", recipe={recipe.name} params={recipe.params.model_dump()}"
        f" base_model={cfg.train.base_model or 'none'}"
        f", strategy={cfg.selection.strategy} params={strategy.params.model_dump()}"
        f" k={cfg.selection.k} needs={sorted(strategy.requires) or 'nothing'}"
    )
    return 0


def cmd_check_release(args: argparse.Namespace) -> int:
    """Validate every configs/releases/{dataset}.yaml (run from the ops repo
    root; --datasets points at the platform checkout's dataset configs)."""
    paths = sorted(RELEASES_DIR.glob("*.yaml"))
    if not paths:
        print(f"no release files in {RELEASES_DIR}")
        return 0
    failed = False
    for path in paths:
        try:
            name, release = ReleaseFile.from_yaml(path)
        except (OSError, ValidationError) as e:
            print(f"{path}: {e}")
            failed = True
            continue
        dataset = Path(args.datasets) / f"{name}.yaml"
        if not dataset.is_file():
            print(f"{path}: no dataset {name} ({dataset} does not exist)")
            failed = True
            continue
        print(f"ok: {name} -> {release.model}/{release.version}")
    return 1 if failed else 0


def cmd_release_input(args: argparse.Namespace) -> int:
    """Print the model-release execution input for one release file (the
    release workflow passes it to StartExecution)."""
    try:
        name, release = ReleaseFile.from_yaml(args.file)
        ev = ReleaseExecution(
            dataset=name,
            model=release.model,
            version=release.version,
            released_by=args.released_by,
            commit_sha=args.commit_sha,
        )
    except (OSError, ValidationError) as e:
        print(e, file=sys.stderr)
        return 1
    print(ev.model_dump_json())
    return 0


def cmd_round_input(args: argparse.Namespace) -> int:
    """Print the start-round Lambda's input for a round config (the start-round
    workflow invokes the Lambda with it)."""
    import yaml

    try:
        with open(args.file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except OSError as e:
        print(e, file=sys.stderr)
        return 1
    payload = {
        "config": config,
        "config_path": args.file,
        "image": args.image,
        "started_by": args.started_by,
        "commit_sha": args.commit_sha,
    }
    print(json.dumps(payload))
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
            images, splits, initial, conflicting = plan(cfg, Path(args.source))
            print(f"dry run: {len(images)} images for {cfg.name}, classes {cfg.codes}")
            if conflicting:
                print(f"  left out {conflicting} images filed under two classes")
            _print_split(split_counts(images, splits))
            print(f"  initial labeled train images (v0): {len(initial)}")
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
        f"initial labeled train {report.initial_labeled}; v0 labels added {report.v0_added}"
        + (
            f"; left out {report.conflicting_dropped} images filed under two classes"
            if report.conflicting_dropped
            else ""
        )
        + (
            f", kept earlier split for {report.kept_existing_split}"
            if report.kept_existing_split
            else ""
        )
    )
    _print_split(report.split_counts)
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    from cli import logs

    if not args.id and not args.errors:
        args.parser.error("give an id to follow, or --errors")
    end = datetime.now(UTC)
    start = end - logs.parse_since(args.since)
    client = logs.client()
    if args.errors:
        rows = logs.recent_errors(client, start, end)
    else:
        rows = logs.follow(client, args.id, start, end)
    for row in rows:
        print(logs.format_row(row))
    if not rows:
        print(f"no log lines in the last {args.since}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="al", description="Active-learning operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("strategies", help="list strategies, required inputs and parameters")
    sub.add_parser("recipes", help="list training recipes and their parameters")
    check = sub.add_parser("check-round", help="validate a round config YAML")
    check.add_argument("config")
    check.add_argument("--datasets", default="configs/datasets", help="folder of dataset configs")
    check_release = sub.add_parser("check-release", help="validate configs/releases/*.yaml")
    check_release.add_argument(
        "--datasets", default="configs/datasets", help="folder of dataset configs"
    )
    round_input = sub.add_parser(
        "round-input", help="print the start-round input for a round config"
    )
    round_input.add_argument("file", help="configs/rounds/<name>.yaml")
    round_input.add_argument("--started-by", required=True, help="GitHub login of the merger")
    round_input.add_argument("--commit-sha", required=True)
    round_input.add_argument("--image", required=True, help="worker image built for the commit")
    release_input = sub.add_parser(
        "release-input", help="print the model-release input for a release file"
    )
    release_input.add_argument("file", help="configs/releases/<dataset>.yaml")
    release_input.add_argument("--released-by", required=True, help="GitHub login of the merger")
    release_input.add_argument("--commit-sha", required=True)

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

    logs = sub.add_parser(
        "logs",
        help="print every /leaf-al/ log line about a round, job, release or request",
        description="Needs AWS credentials (AWS_PROFILE=leaf-al).",
    )
    logs.add_argument("id", nargs="?", help="a round_id, job id, dataset, commit or request id")
    logs.add_argument("--since", default="7d", help="how far back: 30m, 6h, 7d (default 7d)")
    logs.add_argument("--errors", action="store_true", help="recent errors from every part")
    logs.set_defaults(parser=logs)

    args = parser.parse_args(argv)
    if args.command == "db":
        return {"migrate": cmd_db_migrate}[args.db_command](args)
    handlers = {
        "strategies": cmd_strategies,
        "recipes": cmd_recipes,
        "check-round": cmd_check_round,
        "check-release": cmd_check_release,
        "release-input": cmd_release_input,
        "round-input": cmd_round_input,
        "seed": cmd_seed,
        "logs": cmd_logs,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
