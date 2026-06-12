"""Train SimpleBaseline on COCO val2017.

Run from the repository root:
    python scripts/train.py --config configs/coco_simplebaseline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, output_path, project_path
from src.data.coco_dataset import build_coco_datasets
from src.models.simplebaseline import SimpleBaselinePoseNet
from src.training.train import train_model
from src.visualization.visualize import plot_training_curves


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/coco_simplebaseline.yaml",
        help="YAML configuration path relative to the repository root.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. 'auto' uses CUDA when available.",
    )
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    """Resolve the requested device and fail clearly when CUDA is unavailable."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def main() -> None:
    """Load data and train the configured SimpleBaseline model."""
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    loader_config = config["loader"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]
    device = select_device(args.device)

    output_dir = output_path(output_config["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset = build_coco_datasets(
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=True,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=False,
        num_workers=loader_config["num_workers"],
        pin_memory=loader_config["pin_memory"],
    )

    model = SimpleBaselinePoseNet(
        num_joints=model_config["num_joints"],
        pretrained=model_config["pretrained"],
    )
    print(
        f"Device: {device} | Train: {len(train_dataset):,} samples | "
        f"Val: {len(val_dataset):,} samples"
    )
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=training_config["epochs"],
        head_learning_rate=training_config["head_learning_rate"],
        backbone_learning_rate=training_config["backbone_learning_rate"],
        unfreeze_epoch=training_config["unfreeze_epoch"],
        scheduler_step_size=training_config["scheduler_step_size"],
        scheduler_gamma=training_config["scheduler_gamma"],
    )

    weights_path = output_path(output_dir / output_config["weights"])
    history_path = output_path(output_dir / output_config["history"])
    curves_path = output_path(output_dir / output_config["loss_curves"])
    torch.save(model.state_dict(), weights_path)
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)
    plot_training_curves(
        history,
        curves_path,
        unfreeze_epoch=training_config["unfreeze_epoch"],
    )

    print(f"Weights saved to {weights_path}")
    print(f"Training history saved to {history_path}")
    print(f"Loss curves saved to {curves_path}")


if __name__ == "__main__":
    main()
