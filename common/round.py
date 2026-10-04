"""Human-written settings for an AL round.

The owner explores the data, then picks the strategy, its parameters and k for
each round in a YAML file passed to the round CLI. Nothing about selection is
hardcoded in the pipeline. The chosen values are stored on the `rounds` row
(strategy, strategy_cfg, k) so every round is reproducible.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from common.types import DatasetName


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectionConfig(_Strict):
    strategy: str  # a name registered in selection.STRATEGIES
    params: dict[str, Any] = Field(default_factory=dict)  # validated by the strategy
    k: int = Field(gt=0)
    seed: int = 42


class RoundConfig(_Strict):
    dataset: DatasetName
    mode: Literal["simulation", "production"] = "simulation"
    train_config: str  # path to a TrainConfig YAML
    selection: SelectionConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RoundConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))
