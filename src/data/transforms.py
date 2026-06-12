"""Image preprocessing helpers for the pose estimation dataset."""

from typing import Sequence

import torch
import torchvision.transforms.functional as functional
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_resize_normalize(
    image: Image.Image,
    bbox: Sequence[float],
    input_size: Sequence[int] = (256, 192),
) -> torch.Tensor:
    """Crop one person, resize to ``(height, width)``, and normalize.

    Bounding-box conversion intentionally follows the original notebook:
    floating-point COCO values are converted to integers before cropping.
    """
    box_x, box_y, box_width, box_height = bbox
    box_x = int(box_x)
    box_y = int(box_y)
    box_width = max(int(box_width), 1)
    box_height = max(int(box_height), 1)

    input_height, input_width = int(input_size[0]), int(input_size[1])
    cropped = image.crop(
        (box_x, box_y, box_x + box_width, box_y + box_height)
    )
    resized = cropped.resize((input_width, input_height))
    tensor = functional.to_tensor(resized)
    return functional.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display and return a CPU tensor."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype).view(3, 1, 1)
    return (image_tensor.detach().cpu() * std + mean).clamp(0.0, 1.0)
