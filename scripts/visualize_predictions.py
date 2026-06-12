"""Visualize predictions from trained SimpleBaseline weights.

Run from the repository root:
    python scripts/visualize_predictions.py \
        --config configs/coco_simplebaseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, output_path, project_path
from src.data.coco_dataset import build_coco_datasets
from src.models.simplebaseline import SimpleBaselinePoseNet, load_model_weights
from src.visualization.visualize import visualize_predictions


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
        help="Inference device. 'auto' uses CUDA when available.",
    )
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    """Resolve the requested inference device."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def main() -> None:
    """Load validation images and save a grid of model predictions."""
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    visualization_config = config["visualization"]
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

    figure_path = output_path(output_dir / output_config["predictions"])
    visualize_predictions(
        model=model,
        dataset=val_dataset,
        device=device,
        output_path=figure_path,
        num_samples=visualization_config["num_samples"],
        seed=visualization_config["seed"],
        confidence_threshold=visualization_config[
            "confidence_threshold"
        ],
    )
    print(f"Prediction visualization saved to {figure_path}")


if __name__ == "__main__":
    main()
