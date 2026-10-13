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
    cfg = RoundConfig.from_yaml("configs/rounds/example.yaml")
    assert cfg.selection.strategy == "random"
    assert cfg.selection.k == 50


def test_strategies_command(capsys):
    assert main(["strategies"]) == 0
    assert '"random"' in capsys.readouterr().out


def test_check_round_ok(capsys):
    assert main(["check-round", "configs/rounds/example.yaml"]) == 0
    assert capsys.readouterr().out.startswith("ok:")


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
    text = (ROOT / "configs/rounds/example.yaml").read_text()
    assert old in text
    text = text.replace(old, new)
    bad = tmp_path / "round.yaml"
    bad.write_text(text)
    assert main(["check-round", os.fspath(bad)]) == 1
    assert message in capsys.readouterr().out
