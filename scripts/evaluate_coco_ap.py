"""Evaluate SimpleBaseline with COCO OKS keypoint AP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, output_path, project_path
from src.data.coco_dataset import build_coco_datasets
from src.device import select_device
from src.evaluation.coco_eval import (
    evaluate_coco_keypoint_ap,
    predict_coco_keypoints,
)
from src.models.simplebaseline import SimpleBaselinePoseNet, load_model_weights


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


def main() -> None:
    """Run COCO AP evaluation for the validation split."""
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    visualization_config = config["visualization"]
    output_config = config["output"]
    device = select_device(args.device)

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

    output_dir = output_path(output_config["dir"])
    results_dir = output_path(output_config.get("results_dir", output_dir))
    results_dir.mkdir(parents=True, exist_ok=True)
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

    predictions = predict_coco_keypoints(
        model=model,
        dataset=val_dataset,
        device=device,
        heatmap_size=data_config["heatmap_size"],
        confidence_threshold=visualization_config["confidence_threshold"],
    )
    predictions_path = output_path(results_dir / output_config["coco_predictions"])
    metrics = evaluate_coco_keypoint_ap(
        val_dataset.coco,
        predictions,
        predictions_path,
    )
    metrics_path = output_path(results_dir / output_config["coco_metrics"])
    payload = {
        "method": "simplebaseline",
        "iou_type": "keypoints",
        "prediction_file": str(predictions_path.relative_to(PROJECT_ROOT)),
        **metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2)

    print(
        "COCO keypoint AP: "
        + " | ".join(f"{name}: {value:.3f}" for name, value in metrics.items())
    )
    print(f"COCO predictions saved to {predictions_path}")
    print(f"COCO AP metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
