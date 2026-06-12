"""Evaluate trained SimpleBaseline weights with PCK.

Run from the repository root:
    python scripts/evaluate_pck.py --config configs/coco_simplebaseline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, output_path, project_path
from src.data.coco_dataset import build_coco_datasets
from src.evaluation.pck import evaluate_pck
from src.models.simplebaseline import SimpleBaselinePoseNet, load_model_weights
from src.visualization.visualize import JOINT_NAMES, plot_pck_scores


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/coco_simplebaseline.yaml",
        help="YAML configuration path relative to the repository root.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional weights path; defaults to the configured output file.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Evaluation device. 'auto' uses CUDA when available.",
    )
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    """Resolve the requested evaluation device."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def main() -> None:
    """Load the validation split, weights, and report per-joint PCK."""
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    evaluation_config = config["evaluation"]
    output_config = config["output"]
    device = select_device(args.device)

    output_dir = output_path(output_config["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _, val_dataset = build_coco_datasets(
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

    weights_path = (
        project_path(args.weights)
        if args.weights
        else output_dir / output_config["weights"]
    )
    model = SimpleBaselinePoseNet(
        num_joints=model_config["num_joints"],
        pretrained=False,
    ).to(device)
    load_model_weights(model, weights_path, device)
    results = evaluate_pck(
        model=model,
        dataset=val_dataset,
        device=device,
        threshold=evaluation_config["pck_threshold"],
        visibility_threshold=evaluation_config["visibility_threshold"],
        heatmap_size=data_config["heatmap_size"],
    )

    mean_pck = float(results["mean_pck"])
    per_joint_pck = np.asarray(results["per_joint_pck"])
    print(f"SimpleBaseline mean PCK: {mean_pck * 100:.1f}%")
    for joint_name, score in zip(JOINT_NAMES, per_joint_pck):
        if not np.isnan(score):
            print(f"  {joint_name:<15} {score * 100:5.1f}%")

    metrics_path = output_path(output_dir / output_config["pck_results"])
    chart_path = output_path(output_dir / output_config["pck_chart"])
    serializable_results = {
        "mean_pck": mean_pck,
        "per_joint_pck": {
            name: None if np.isnan(score) else float(score)
            for name, score in zip(JOINT_NAMES, per_joint_pck)
        },
        "threshold": evaluation_config["pck_threshold"],
    }
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(serializable_results, metrics_file, indent=2)
    plot_pck_scores(per_joint_pck, mean_pck, chart_path)

    print(f"PCK results saved to {metrics_path}")
    print(f"PCK chart saved to {chart_path}")


if __name__ == "__main__":
    main()
