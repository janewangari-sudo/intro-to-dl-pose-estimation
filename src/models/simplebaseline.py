"""ResNet-50 SimpleBaseline model from the reference notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


def _deconv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    """Build one SimpleBaseline deconvolution, batch norm, and ReLU block."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class SimpleBaselinePoseNet(nn.Module):
    """Predict one heatmap per joint from a cropped person image.

    The architecture is unchanged from the notebook: an ImageNet-pretrained
    ResNet-50 backbone followed by three 256-channel deconvolution blocks and
    a final 1x1 convolution.
    """

    def __init__(
        self,
        num_joints: int = 17,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.head = nn.Sequential(
            _deconv_block(2048, 256),
            _deconv_block(256, 256),
            _deconv_block(256, 256),
            nn.Conv2d(256, num_joints, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return heatmaps with shape ``(batch, joints, height, width)``."""
        return self.head(self.backbone(images))


def load_model_weights(
    model: nn.Module,
    weights_path: str | Path,
    device: torch.device | str,
) -> None:
    """Load a raw state dict or a checkpoint containing ``state_dict``."""
    path = Path(weights_path)
    try:
        checkpoint: Any = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
