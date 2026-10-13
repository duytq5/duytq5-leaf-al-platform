from pydantic import Field

from common.manifest import Preprocess
from recipes.base import RecipeParams, TrainRecipe, register

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

    @property
    def arch(self) -> str:
        return self.params.arch

    def model_name(self, dataset: str) -> str:
        # fastvit_t8 -> rocole-fastvit: one registered model per family.
        return f"{dataset}-{self.params.arch.split('_')[0]}"

    def preprocess(self) -> Preprocess:
        size = self.params.input_size
        return Preprocess(input_size=(size, size), mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def build_model(self, num_classes, base):
        import timm

        if base is None:
            return timm.create_model(
                self.params.arch, pretrained=self.params.pretrained, num_classes=num_classes
            )
        if base.num_classes != num_classes:  # the dataset gained a class
            base.reset_classifier(num_classes)
        return base

    def export_model(self, model):
        from timm.utils.model import reparameterize_model

        # FastViT and other re-parameterizable models fold their branches.
        return reparameterize_model(model)

    def train_transforms(self):
        from torchvision import transforms

        pre = self.preprocess()
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(pre.input_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(pre.mean, pre.std),
            ]
        )

    def configure_optimizer(self, model):
        import torch

        return torch.optim.AdamW(
            model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay
        )
