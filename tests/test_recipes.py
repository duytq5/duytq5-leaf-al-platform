"""Training recipe registry (design doc change 16, F6).

A recipe is a registered TrainRecipe subclass with a Params model; round
configs name it in `train: {recipe, params, base_model}`. Looking recipes up
and checking params needs no torch, so CI and `al check-round` run anywhere.
"""

import subprocess
import sys

import pytest
from pydantic import ValidationError

from recipes import RecipeParams, TrainRecipe, describe_recipes, get_recipe, register


def test_default_recipe_is_registered():
    recipe = get_recipe("timm_classifier", {})
    assert recipe.name == "timm_classifier"
    assert recipe.params.arch == "fastvit_t8"


def test_params_are_validated_and_typos_fail():
    recipe = get_recipe("timm_classifier", {"epochs": 3, "lr": "1e-3"})
    assert (recipe.params.epochs, recipe.params.lr) == (3, 1e-3)
    with pytest.raises(ValidationError):
        get_recipe("timm_classifier", {"epoch": 3})
    with pytest.raises(ValidationError):
        get_recipe("timm_classifier", {"epochs": 0})


def test_unknown_recipe():
    with pytest.raises(KeyError, match="unknown recipe 'nope'"):
        get_recipe("nope", {})


def test_describe_lists_params():
    described = describe_recipes()
    params = described["timm_classifier"]["params"]["properties"]
    assert {"arch", "epochs", "batch_size", "lr"} <= set(params)
    assert described["timm_classifier"]["description"]


def test_a_name_registers_once():
    with pytest.raises(ValueError, match="already registered"):

        @register
        class Again(TrainRecipe):
            name = "timm_classifier"

            class Params(RecipeParams):
                pass


def test_recipe_params_reject_unknown_keys_by_default():
    class P(RecipeParams):
        x: int = 1

    with pytest.raises(ValidationError):
        P.model_validate({"y": 2})


def test_looking_up_recipes_does_not_import_torch():
    code = (
        "import sys, recipes; recipes.get_recipe('timm_classifier', {}); "
        "assert 'torch' not in sys.modules and 'timm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
