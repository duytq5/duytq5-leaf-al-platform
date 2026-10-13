"""Training recipes; importing this package registers the built-in ones."""

from recipes import timm_classifier  # noqa: F401  (registers the recipe)
from recipes.base import RecipeParams, TrainRecipe, describe_recipes, get_recipe, register

__all__ = ["RecipeParams", "TrainRecipe", "describe_recipes", "get_recipe", "register"]
