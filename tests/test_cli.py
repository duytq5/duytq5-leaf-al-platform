import os
from pathlib import Path

import pytest

from cli.main import main
from common.round import RoundConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_example_round_config_loads():
    cfg = RoundConfig.from_yaml("configs/examples/round.yaml")
    assert cfg.selection.strategy == "random"
    assert cfg.selection.k == 50


def test_strategies_command(capsys):
    assert main(["strategies"]) == 0
    assert '"random"' in capsys.readouterr().out


def test_check_round_ok(capsys):
    assert main(["check-round", "configs/examples/round.yaml"]) == 0
    assert capsys.readouterr().out.startswith("ok:")


def test_check_round_without_train(tmp_path, capsys):
    """Change 24: a selection-only round, with or without its scoring model."""
    text = (ROOT / "configs/examples/round.yaml").read_text()
    head, rest = text.split("train:\n", 1)
    selection_only = head + "selection:" + rest.split("selection:", 1)[1]
    path = tmp_path / "round.yaml"
    # The example's strategy is random: it needs no scores, so no model.
    path.write_text(selection_only)
    assert main(["check-round", os.fspath(path)]) == 0
    assert "no training, no scoring" in capsys.readouterr().out
    path.write_text(selection_only + "model: models:/rocole-fastvit@champion\n")
    assert main(["check-round", os.fspath(path)]) == 0
    assert "scores with models:/rocole-fastvit@champion" in capsys.readouterr().out
    path.write_text(selection_only.replace("strategy: random", "strategy: margin"))
    assert main(["check-round", os.fspath(path)]) == 0
    assert "no training, scores with the newest model" in capsys.readouterr().out


def test_check_training(tmp_path, capsys):
    """Change 24: the ops repo's CI checks configs/trainings/*.yaml."""
    assert main(["check-training", "configs/examples/training.yaml"]) == 0
    assert capsys.readouterr().out.startswith("ok: training on rocole latest version")
    bad = tmp_path / "t.yaml"
    bad.write_text("dataset: rocole\ntrain: {recipe: nope}\n")
    assert main(["check-training", os.fspath(bad)]) == 1
    assert "unknown recipe" in capsys.readouterr().out
    bad.write_text("dataset: pepper\ntrain: {recipe: timm_classifier}\n")
    assert main(["check-training", os.fspath(bad)]) == 1
    assert "no dataset pepper" in capsys.readouterr().out


def test_recipes_command(capsys):
    assert main(["recipes"]) == 0
    assert '"timm_classifier"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("strategy: random", "strategy: nope", "unknown strategy"),
        ("params: {}  # selection", "params: {x: 1}", "Extra inputs"),
        ("recipe: timm_classifier", "recipe: nope", "unknown recipe"),
        ("params: {epochs: 20}", "params: {epoch: 20}", "Extra inputs"),
        ("dataset: rocole", "dataset: pepper", "configs/datasets/pepper.yaml"),
        ("k: 50", "k: 0", "greater than 0"),
        ("dataset: rocole", "dataset: rocole\nmode: simulation", "Extra inputs"),
    ],
)
def test_check_round_rejects(tmp_path, capsys, old, new, message):
    text = (ROOT / "configs/examples/round.yaml").read_text()
    assert old in text
    text = text.replace(old, new)
    bad = tmp_path / "round.yaml"
    bad.write_text(text)
    assert main(["check-round", os.fspath(bad)]) == 1
    assert message in capsys.readouterr().out


def test_check_round_from_the_ops_repo(tmp_path, monkeypatch, capsys):
    """The ops repo's CI checks its round files with the pinned platform's
    `al` and the platform checkout's dataset configs."""
    rounds = tmp_path / "ops/configs/rounds"
    rounds.mkdir(parents=True)
    (rounds / "r1.yaml").write_text((ROOT / "configs/examples/round.yaml").read_text())
    datasets = tmp_path / "platform/configs/datasets"
    datasets.mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "ops")
    args = ["check-round", "configs/rounds/r1.yaml", "--datasets", str(datasets)]
    assert main(args) == 1
    assert "no dataset rocole" in capsys.readouterr().out
    (datasets / "rocole.yaml").write_text("name: rocole\n")
    assert main(args) == 0
