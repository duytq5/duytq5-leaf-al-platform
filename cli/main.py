import argparse
import json
import sys

from pydantic import ValidationError

from common.config import TrainConfig
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="al", description="Active-learning operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("strategies", help="list strategies, required inputs and parameters")
    check = sub.add_parser("check-round", help="validate a round config YAML")
    check.add_argument("config")
    args = parser.parse_args(argv)
    handlers = {"strategies": cmd_strategies, "check-round": cmd_check_round}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
