"""Percentage of Correct Keypoints (PCK) evaluation."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def heatmaps_to_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    """Convert ``(joints, height, width)`` heatmaps to peak ``(x, y)`` points."""
    if heatmaps.ndim != 3:
        raise ValueError(
            "Expected heatmaps with shape (joints, height, width), "
            f"received {tuple(heatmaps.shape)}"
        )

    num_joints, _, heatmap_width = heatmaps.shape
    peak_indices = heatmaps.reshape(num_joints, -1).argmax(dim=1)
    return torch.stack(
        [peak_indices % heatmap_width, peak_indices // heatmap_width],
        dim=1,
    ).float()


def compute_pck(
    predicted_coords: np.ndarray,
    target_coords: np.ndarray,
    target_visibility: np.ndarray,
    threshold: float = 0.2,
    heatmap_size: Sequence[int] = (64, 48),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-joint correct/visible indicators for one person crop.

    As in the notebook, the correctness radius is a fraction of the heatmap
    diagonal rather than a torso- or head-normalized distance.
    """
    heatmap_height, heatmap_width = heatmap_size
    radius = threshold * np.sqrt(heatmap_height**2 + heatmap_width**2)
    num_joints = predicted_coords.shape[0]
    correct = np.zeros(num_joints, dtype=np.float64)
    visible = np.zeros(num_joints, dtype=np.float64)

    for joint_index in range(num_joints):
        if target_visibility[joint_index] == 0:
            continue
        visible[joint_index] = 1
        distance = np.linalg.norm(
            predicted_coords[joint_index] - target_coords[joint_index]
        )
        if distance <= radius:
            correct[joint_index] = 1

    return correct, visible


def evaluate_pck(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device | str,
    threshold: float = 0.2,
    visibility_threshold: float = 0.01,
    heatmap_size: Sequence[int] = (64, 48),
) -> Dict[str, np.ndarray | float]:
    """Evaluate mean and per-joint PCK over a pose dataset."""
    model.to(device)
    model.eval()
    num_joints = getattr(dataset, "num_joints", 17)
    total_correct = np.zeros(num_joints, dtype=np.float64)
    total_visible = np.zeros(num_joints, dtype=np.float64)

    with torch.no_grad():
        for image, target_heatmaps in dataset:
            predicted_heatmaps = (
                model(image.unsqueeze(0).to(device)).squeeze(0).cpu()
            )
            predicted_coords = heatmaps_to_coords(
                predicted_heatmaps
            ).numpy()
            target_coords = heatmaps_to_coords(target_heatmaps).numpy()
            target_visibility = (
                target_heatmaps.reshape(num_joints, -1).max(dim=1).values
                > visibility_threshold
            ).numpy()

            correct, visible = compute_pck(
                predicted_coords,
                target_coords,
                target_visibility,
                threshold=threshold,
                heatmap_size=heatmap_size,
            )
            total_correct += correct
            total_visible += visible

    per_joint = np.full(num_joints, np.nan, dtype=np.float64)
    np.divide(
        total_correct,
        total_visible,
        out=per_joint,
        where=total_visible > 0,
    )
    return {
        "mean_pck": float(np.nanmean(per_joint)),
        "per_joint_pck": per_joint,
        "correct": total_correct,
        "visible": total_visible,
    }
