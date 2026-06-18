"""COCO OKS-based keypoint AP evaluation helpers."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from pycocotools.cocoeval import COCOeval

from .pck import heatmaps_to_coords


COCO_AP_NAMES = ("AP", "AP50", "AP75", "AP_medium", "AP_large")


def heatmaps_to_coco_keypoints(
    heatmaps: torch.Tensor,
    bbox: Sequence[float],
    heatmap_size: Sequence[int],
    confidence_threshold: float = 0.0,
) -> tuple[list[float], float]:
    """Map predicted heatmap peaks back to COCO image coordinates."""
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must have shape (joints, height, width)")

    bbox_x, bbox_y, bbox_width, bbox_height = [float(value) for value in bbox]
    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError("bbox width and height must be positive")

    heatmap_height, heatmap_width = (
        int(heatmap_size[0]),
        int(heatmap_size[1]),
    )
    coordinates = heatmaps_to_coords(heatmaps).numpy()
    confidence = (
        heatmaps.reshape(heatmaps.shape[0], -1).max(dim=1).values.numpy()
    )

    keypoints: list[float] = []
    visible_confidences: list[float] = []
    for (heatmap_x, heatmap_y), score in zip(coordinates, confidence):
        image_x = bbox_x + float(heatmap_x) / heatmap_width * bbox_width
        image_y = bbox_y + float(heatmap_y) / heatmap_height * bbox_height
        visibility = 2 if float(score) >= confidence_threshold else 0
        keypoints.extend([image_x, image_y, visibility])
        if visibility > 0:
            visible_confidences.append(float(score))

    detection_score = (
        float(np.mean(visible_confidences)) if visible_confidences else 0.0
    )
    return keypoints, detection_score


def predict_coco_keypoints(
    model: nn.Module,
    dataset,
    device: torch.device | str,
    heatmap_size: Sequence[int],
    confidence_threshold: float = 0.0,
) -> list[dict]:
    """Run a pose model over a COCO pose dataset and return result records."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for sample_index in range(len(dataset)):
            image, _ = dataset[sample_index]
            image_id, annotation = dataset.samples[sample_index]
            heatmaps = model(image.unsqueeze(0).to(device)).squeeze(0).cpu()
            keypoints, score = heatmaps_to_coco_keypoints(
                heatmaps,
                annotation["bbox"],
                heatmap_size=heatmap_size,
                confidence_threshold=confidence_threshold,
            )
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(dataset.person_category_id),
                    "keypoints": keypoints,
                    "score": score,
                }
            )

    return predictions


def evaluate_coco_keypoint_ap(
    coco_gt,
    predictions: list[dict],
    output_path: str | Path,
) -> dict[str, float]:
    """Save COCO detections and return AP metrics from COCOeval."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(predictions, output_file, indent=2)

    if not predictions:
        raise ValueError("No predictions available for COCO evaluation.")

    coco_dt = coco_gt.loadRes(str(output_path))
    evaluator = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        name: float(value)
        for name, value in zip(COCO_AP_NAMES, evaluator.stats[:5])
    }
