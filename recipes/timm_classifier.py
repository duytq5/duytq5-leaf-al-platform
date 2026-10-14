from pydantic import Field

from recipes.base import RecipeParams, TrainRecipe, register


@register
class TimmClassifier(TrainRecipe):
    """An image classifier from timm (default FastViT T8, ImageNet weights)
    with a new head for the dataset's classes, trained with AdamW."""

    name = "timm_classifier"

    class Params(RecipeParams):
        arch: str = "fastvit_t8"
        pretrained: bool = True  # ImageNet weights; ignored when base_model is set
        input_size: int = Field(default=256, ge=32)
        epochs: int = Field(default=20, gt=0)
        batch_size: int = Field(default=64, gt=0)
        lr: float = Field(default=3e-4, gt=0)  # YAML reads 3e-4 as a string; coerced
        weight_decay: float = Field(default=0.05, ge=0)
        seed: int = 42
