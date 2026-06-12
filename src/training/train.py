"""Training loop for heatmap-based pose estimation."""

from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the model's ResNet backbone."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def _fine_tuning_optimizer(
    model: nn.Module,
    backbone_learning_rate: float,
    head_learning_rate: float,
) -> Adam:
    """Create the two-rate optimizer used after backbone unfreezing."""
    return Adam(
        [
            {
                "params": model.backbone.parameters(),
                "lr": backbone_learning_rate,
            },
            {
                "params": model.head.parameters(),
                "lr": head_learning_rate,
            },
        ]
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | str,
    epochs: int = 30,
    head_learning_rate: float = 1e-3,
    backbone_learning_rate: float = 1e-4,
    unfreeze_epoch: int = 5,
    scheduler_step_size: int = 20,
    scheduler_gamma: float = 0.1,
    log: Callable[[str], None] = print,
) -> Dict[str, List[float]]:
    """Train with MSE heatmap loss and the notebook's two-stage strategy.

    ``unfreeze_epoch`` is zero-based, matching the original loop. The default
    value therefore unfreezes the backbone before the sixth training epoch.
    """
    model.to(device)
    criterion = nn.MSELoss()

    set_backbone_trainable(model, False)
    optimizer = Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=head_learning_rate,
    )
    scheduler = StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            set_backbone_trainable(model, True)
            optimizer = _fine_tuning_optimizer(
                model,
                backbone_learning_rate=backbone_learning_rate,
                head_learning_rate=head_learning_rate,
            )
            # Keep the scheduler attached to the initial optimizer to preserve
            # the exact training sequence used in the reference notebook.

        model.train()
        running_train_loss = 0.0
        for images, target_heatmaps in train_loader:
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), target_heatmaps)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, target_heatmaps in val_loader:
                images = images.to(device)
                target_heatmaps = target_heatmaps.to(device)
                batch_loss = criterion(model(images), target_heatmaps)
                running_val_loss += batch_loss.item() * images.size(0)

        train_loss = running_train_loss / len(train_loader.dataset)
        val_loss = running_val_loss / len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        log(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train: {train_loss:.5f} | val: {val_loss:.5f}"
        )

    return {"train_loss": train_losses, "val_loss": val_losses}
