"""Evaluate a fixed average-pose baseline on the COCO validation split.

Run from the repository root:
    python scripts/evaluate_trivial_baseline.py \
        --config configs/coco_simplebaseline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, output_path, project_path
from src.data.coco_dataset import COCOPoseDataset, build_coco_datasets
from src.evaluation.pck import compute_pck, heatmaps_to_coords
from src.models.trivial_baseline import (
    AveragePoseTemplate,
    estimate_average_pose,
)
from src.visualization.visualize import JOINT_NAMES


TEMPLATE_PATH = "outputs/results/trivial_baseline_template.json"
RESULTS_PATH = "outputs/results/trivial_baseline_pck.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/coco_simplebaseline.yaml",
        help="YAML configuration path relative to the repository root.",
    )
    return parser.parse_args()


def build_datasets(config: dict[str, Any]) -> tuple[
    COCOPoseDataset,
    COCOPoseDataset,
]:
    """Build the same configured train/validation split as model training."""
    data_config = config["data"]
    return build_coco_datasets(
        data_dir=project_path(data_config["dir"]),
        input_size=data_config["input_size"],
        heatmap_size=data_config["heatmap_size"],
        sigma=data_config["sigma"],
        train_fraction=data_config["train_fraction"],
        split_seed=data_config["split_seed"],
        min_image_keypoints=data_config["min_image_keypoints"],
        min_annotation_keypoints=data_config["min_annotation_keypoints"],
        download=data_config["download"],
    )


def evaluate_template(
    template: AveragePoseTemplate,
    val_dataset: COCOPoseDataset,
    heatmap_size: Sequence[int],
    threshold: float,
    visibility_threshold: float,
) -> dict[str, np.ndarray | float | int]:
    """Evaluate a fixed template with the project's existing PCK functions."""
    if len(template.sample_counts) != val_dataset.num_joints:
        raise ValueError(
            "Template and validation dataset must use the same joint count."
        )

    predicted_coordinates = template.to_heatmap_coordinates(heatmap_size)
    total_correct = np.zeros(val_dataset.num_joints, dtype=np.float64)
    total_visible = np.zeros(val_dataset.num_joints, dtype=np.float64)

    for sample_index in range(len(val_dataset)):
        _, target_heatmaps = val_dataset[sample_index]
        target_coordinates = heatmaps_to_coords(target_heatmaps).numpy()
        target_visibility = (
            target_heatmaps.reshape(val_dataset.num_joints, -1)
            .max(dim=1)
            .values
            .numpy()
            > visibility_threshold
        )
        target_visibility &= template.valid_joints

        correct, visible = compute_pck(
            predicted_coordinates,
            target_coordinates,
            target_visibility,
            threshold=threshold,
            heatmap_size=heatmap_size,
        )
        total_correct += correct
        total_visible += visible

    per_joint_pck = np.full(
        val_dataset.num_joints,
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        total_correct,
        total_visible,
        out=per_joint_pck,
        where=total_visible > 0,
    )
    if not np.any(total_visible > 0):
        raise ValueError("No valid validation keypoints available for PCK.")

    return {
        "mean_pck": float(np.nanmean(per_joint_pck)),
        "per_joint_pck": per_joint_pck,
        "num_valid_keypoints": int(total_visible.sum()),
        "per_joint_valid_keypoints": total_visible.astype(np.int64),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Create the parent directory and save indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)


def relative_project_path(path: Path) -> str:
    """Return a portable repository-relative path for result metadata."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def make_result_payload(
    evaluation: dict[str, np.ndarray | float | int],
    threshold: float,
    template_path: Path,
) -> dict[str, Any]:
    """Convert metric arrays into the required report-friendly JSON schema."""
    per_joint_pck = np.asarray(evaluation["per_joint_pck"])
    per_joint_counts = np.asarray(
        evaluation["per_joint_valid_keypoints"]
    )
    return {
        "method": "trivial_average_pose",
        "mean_pck": float(evaluation["mean_pck"]),
        "per_joint_pck": {
            name: None if np.isnan(score) else float(score)
            for name, score in zip(JOINT_NAMES, per_joint_pck)
        },
        "threshold": float(threshold),
        "num_valid_keypoints": int(evaluation["num_valid_keypoints"]),
        "per_joint_valid_keypoints": {
            name: int(count)
            for name, count in zip(JOINT_NAMES, per_joint_counts)
        },
        "template_path": relative_project_path(template_path),
    }


def main() -> None:
    """Estimate, evaluate, and save the trivial average-pose baseline."""
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    model_config = config["model"]

    train_dataset, val_dataset = build_datasets(config)
    template = estimate_average_pose(
        train_dataset.samples,
        num_joints=model_config["num_joints"],
    )

    template_path = output_path(TEMPLATE_PATH)
    results_path = output_path(RESULTS_PATH)
    save_json(template_path, template.to_dict(JOINT_NAMES))

    evaluation = evaluate_template(
        template=template,
        val_dataset=val_dataset,
        heatmap_size=data_config["heatmap_size"],
        threshold=evaluation_config["pck_threshold"],
        visibility_threshold=evaluation_config["visibility_threshold"],
    )
    result_payload = make_result_payload(
        evaluation,
        threshold=evaluation_config["pck_threshold"],
        template_path=template_path,
    )
    save_json(results_path, result_payload)

    print(
        "Trivial average-pose mean PCK: "
        f"{result_payload['mean_pck'] * 100:.1f}%"
    )
    for joint_name, score in result_payload["per_joint_pck"].items():
        if score is not None:
            print(f"  {joint_name:<15} {score * 100:5.1f}%")
    print(f"Template saved to {template_path}")
    print(f"PCK results saved to {results_path}")


if __name__ == "__main__":
    main()
