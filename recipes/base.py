"""Training recipe registry (design doc change 16, F6).

The platform owns the training cycle (data, validation, checkpoints, test
evaluation, MLflow); a recipe owns the model. A recipe is a TrainRecipe
subclass registered under a name, with a `Params` model holding what a round
config may set in `train.params`. Adding a model is one new registered class.

Looking a recipe up and validating its params needs no torch: recipes import
torch and timm inside their methods, so CI and `al check-round` run without
them. The worker calls the training hooks.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class RecipeParams(BaseModel):
    """Base for a recipe's parameters. Unknown keys are rejected, so a typo in a
    round config fails instead of silently training with a default."""

    model_config = ConfigDict(extra="forbid")


class TrainRecipe:
    name: ClassVar[str]
    Params: ClassVar[type[RecipeParams]] = RecipeParams

    def __init__(self, params: RecipeParams) -> None:
        self.params = params


_RECIPES: dict[str, type[TrainRecipe]] = {}


def register(cls: type[TrainRecipe]) -> type[TrainRecipe]:
    """Class decorator: make a recipe available under `cls.name`."""
    if cls.name in _RECIPES:
        raise ValueError(f"recipe {cls.name!r} is already registered")
    _RECIPES[cls.name] = cls
    return cls


def get_recipe(name: str, params: dict[str, Any]) -> TrainRecipe:
    """The recipe `name` with validated params; KeyError for an unknown name."""
    try:
        cls = _RECIPES[name]
    except KeyError:
        raise KeyError(f"unknown recipe {name!r}; known: {sorted(_RECIPES)}") from None
    return cls(cls.Params.model_validate(params))


def describe_recipes() -> dict[str, dict[str, Any]]:
    """Every recipe with its description and parameter schema (`al recipes`)."""
    return {
        name: {
            "description": " ".join((cls.__doc__ or "").split()),
            "params": cls.Params.model_json_schema(),
        }
        for name, cls in sorted(_RECIPES.items())
    }
