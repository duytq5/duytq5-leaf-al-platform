"""Training recipe registry (design doc change 16, F6).

The platform owns the training cycle (data, validation, checkpoints, test
evaluation, MLflow); a recipe owns the model. A recipe is a TrainRecipe
subclass registered under a name, with a `Params` model holding what a round
config may set in `train.params`. Adding a model is one new registered class.

Looking a recipe up and validating its params needs no torch: recipes import
torch and timm inside their methods, so CI and `al check-round` run without
them. The worker calls the training hooks.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from common.manifest import Preprocess

if TYPE_CHECKING:
    import torch


class RecipeParams(BaseModel):
    """Base for a recipe's parameters. Unknown keys are rejected, so a typo in a
    round config fails instead of silently training with a default."""

    model_config = ConfigDict(extra="forbid")

    # The platform's training cycle reads these.
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=64, gt=0)
    seed: int = 42


class TrainContext:
    """What the platform hands `fit`: the data and the end-of-epoch hook, which
    evaluates on val, logs to MLflow and keeps the best checkpoint."""

    def __init__(
        self,
        *,
        train_loader,
        device: str,
        autocast: Callable[[], Any],
        end_epoch: Callable[[int, "torch.nn.Module", float], None],
    ) -> None:
        self.train_loader, self.device = train_loader, device
        self.autocast, self.end_epoch = autocast, end_epoch


class TrainRecipe:
    """A model family. Subclasses set `name` and `Params` and implement
    `build_model` and `preprocess`; the other hooks have working defaults
    (cross-entropy, AdamW, plain epochs) and can be overridden."""

    name: ClassVar[str]
    Params: ClassVar[type[RecipeParams]] = RecipeParams

    def __init__(self, params: RecipeParams) -> None:
        self.params = params

    # What the platform and the app need to know.

    @property
    def arch(self) -> str:
        return self.name

    def model_name(self, dataset: str) -> str:
        """The MLflow registered model a round of `dataset` adds a version to."""
        return f"{dataset}-{self.name}"

    def preprocess(self) -> Preprocess:
        """Input size and normalization, copied into the app's manifest."""
        raise NotImplementedError

    # The model.

    def build_model(self, num_classes: int, base: "torch.nn.Module | None") -> "torch.nn.Module":
        """A model with `num_classes` outputs, in class order. `base` is the
        round's base_model loaded from MLflow, to fine-tune from."""
        raise NotImplementedError

    def export_model(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """The model as exported to ONNX (e.g. reparameterized); same outputs."""
        return model

    def train_transforms(self) -> Callable:
        return self.eval_transforms()

    def eval_transforms(self) -> Callable:
        """PIL image to a normalized CHW tensor, as the app preprocesses."""
        from torchvision import transforms

        pre = self.preprocess()
        return transforms.Compose(
            [
                transforms.Resize(pre.input_size),
                transforms.ToTensor(),
                transforms.Normalize(pre.mean, pre.std),
            ]
        )

    # Training.

    def configure_optimizer(self, model: "torch.nn.Module"):
        import torch

        return torch.optim.AdamW(model.parameters())

    def loss_fn(self) -> Callable:
        import torch

        return torch.nn.CrossEntropyLoss()

    def train_step(self, model, batch, optimizer, loss_fn) -> float:
        """One optimizer step on one batch of (images, labels)."""
        images, labels = batch
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def fit(self, model: "torch.nn.Module", ctx: TrainContext) -> None:
        """Run the epochs; call ctx.end_epoch after each one."""
        optimizer = self.configure_optimizer(model)
        loss_fn = self.loss_fn()
        for epoch in range(self.params.epochs):
            model.train()
            losses = []
            for images, labels in ctx.train_loader:
                batch = (images.to(ctx.device), labels.to(ctx.device))
                with ctx.autocast():
                    losses.append(self.train_step(model, batch, optimizer, loss_fn))
            ctx.end_epoch(epoch, model, sum(losses) / max(len(losses), 1))


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
