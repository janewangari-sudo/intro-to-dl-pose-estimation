"""Visualization functions extracted from the reference notebook."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from PIL import Image
from torch.utils.data import Dataset

from ..data.transforms import denormalize_image
from ..evaluation.pck import heatmaps_to_coords


SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
LIMB_COLORS = (
    ["#FF6B6B"] * 4
    + ["#4ECDC4"] * 5
    + ["#FFE66D"] * 3
    + ["#A8E6CF"] * 4
)
JOINT_NAMES = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]


def draw_ground_truth_pose(
    axis: Axes,
    image: Image.Image,
    annotations: Iterable[dict],
) -> None:
    """Draw visible COCO ground-truth joints and limbs on an image."""
    axis.imshow(image)
    axis.axis("off")

    for annotation in annotations:
        keypoints = np.asarray(annotation["keypoints"]).reshape(-1, 3)
        for x_coord, y_coord, visibility in keypoints:
            if visibility > 0:
                axis.plot(
                    x_coord,
                    y_coord,
                    "o",
                    markersize=5,
                    color="white",
                    zorder=3,
                )
                axis.plot(
                    x_coord,
                    y_coord,
                    "o",
                    markersize=3,
                    color="#FF6B6B",
                    zorder=4,
                )

        for (start, end), color in zip(SKELETON, LIMB_COLORS):
            start_x, start_y, start_visibility = keypoints[start]
            end_x, end_y, end_visibility = keypoints[end]
            if start_visibility > 0 and end_visibility > 0:
                axis.plot(
                    [start_x, end_x],
                    [start_y, end_y],
                    "-",
                    linewidth=2,
                    color=color,
                    zorder=2,
                )


def draw_predictions(
    axis: Axes,
    image_tensor: torch.Tensor,
    predicted_heatmaps: torch.Tensor,
    confidence_threshold: float = 0.1,
) -> None:
    """Draw predicted joints and skeleton on a normalized person crop."""
    image = denormalize_image(image_tensor).permute(1, 2, 0).numpy()
    image_height, image_width = image.shape[:2]
    _, heatmap_height, heatmap_width = predicted_heatmaps.shape
    confidence = (
        predicted_heatmaps.reshape(predicted_heatmaps.shape[0], -1)
        .max(dim=1)
        .values
    )
    coordinates = heatmaps_to_coords(predicted_heatmaps)
    coordinates[:, 0] *= image_width / heatmap_width
    coordinates[:, 1] *= image_height / heatmap_height

    axis.imshow(image)
    axis.axis("off")
    for joint_index, (x_coord, y_coord) in enumerate(coordinates):
        if confidence[joint_index] > confidence_threshold:
            axis.plot(
                x_coord.item(),
                y_coord.item(),
                "o",
                markersize=6,
                color="white",
                zorder=3,
            )
            axis.plot(
                x_coord.item(),
                y_coord.item(),
                "o",
                markersize=4,
                color="#FF6B6B",
                zorder=4,
            )

    for (start, end), color in zip(SKELETON, LIMB_COLORS):
        if (
            confidence[start] > confidence_threshold
            and confidence[end] > confidence_threshold
        ):
            start_x, start_y = coordinates[start]
            end_x, end_y = coordinates[end]
            axis.plot(
                [start_x.item(), end_x.item()],
                [start_y.item(), end_y.item()],
                "-",
                linewidth=2,
                color=color,
                zorder=2,
            )


def plot_training_curves(
    history: dict[str, Sequence[float]],
    output_path: str | Path,
    unfreeze_epoch: int = 5,
) -> None:
    """Save train and validation MSE loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(
        epochs,
        history["train_loss"],
        label="Train",
        color="#4ECDC4",
        linewidth=2,
    )
    axis.plot(
        epochs,
        history["val_loss"],
        label="Val",
        color="#FF6B6B",
        linewidth=2,
        linestyle="--",
    )
    axis.axvline(
        x=unfreeze_epoch + 1,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
        label="Backbone unfreeze",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE Loss")
    axis.set_title("SimpleBaseline training and validation loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_pck_scores(
    per_joint_pck: np.ndarray,
    mean_pck: float,
    output_path: str | Path,
    joint_names: Sequence[str] = JOINT_NAMES,
) -> None:
    """Save the notebook-style per-joint PCK bar chart."""
    valid_mask = ~np.isnan(per_joint_pck)
    valid_names = [
        name for name, is_valid in zip(joint_names, valid_mask) if is_valid
    ]
    valid_scores = per_joint_pck[valid_mask]
    colors = [
        "#4ECDC4" if score >= mean_pck else "#FF6B6B"
        for score in valid_scores
    ]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        valid_names,
        valid_scores * 100,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    axis.axhline(
        mean_pck * 100,
        color="#FFE66D",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_pck * 100:.1f}%",
    )
    axis.set_ylabel("PCK (%)")
    axis.set_title("SimpleBaseline per-joint PCK on COCO val2017")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def visualize_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device | str,
    output_path: str | Path,
    num_samples: int = 6,
    seed: int = 7,
    confidence_threshold: float = 0.1,
) -> None:
    """Run inference on random validation samples and save a prediction grid."""
    if len(dataset) == 0:
        raise ValueError("Cannot visualize predictions from an empty dataset.")

    sample_count = min(num_samples, len(dataset))
    selected_indices = random.Random(seed).sample(
        range(len(dataset)),
        sample_count,
    )
    columns = min(3, sample_count)
    rows = math.ceil(sample_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
        squeeze=False,
    )
    figure.suptitle(
        "SimpleBaseline predicted keypoints on validation images",
        fontsize=14,
    )

    model.to(device)
    model.eval()
    for axis, sample_index in zip(axes.flat, selected_indices):
        image, _ = dataset[sample_index]
        with torch.no_grad():
            predicted_heatmaps = (
                model(image.unsqueeze(0).to(device)).squeeze(0).cpu()
            )
        draw_predictions(
            axis,
            image,
            predicted_heatmaps,
            confidence_threshold=confidence_threshold,
        )
    for axis in list(axes.flat)[sample_count:]:
        axis.axis("off")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
