"""Train SimpleBaseline on COCO val2017.

Run from the repository root:
    python scripts/train.py --config configs/coco_simplebaseline.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, output_path, project_path
from src.data.coco_dataset import build_coco_datasets
from src.device import select_device
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


def make_data_loader(
    dataset,
    loader_config: dict,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader with GPU-friendly options when available."""
    num_workers = int(loader_config["num_workers"])
    loader_kwargs = {
        "batch_size": loader_config["batch_size"],
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(loader_config["pin_memory"])
        and device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            loader_config.get("persistent_workers", False)
        )
        loader_kwargs["prefetch_factor"] = int(
            loader_config.get("prefetch_factor", 2)
        )
    return DataLoader(dataset, **loader_kwargs)


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
    config_copy_path = output_path(output_dir / output_config["config_copy"])
    shutil.copy2(project_path(args.config), config_copy_path)
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
    train_loader = make_data_loader(
        train_dataset,
        loader_config,
        device=device,
        shuffle=True,
    )
    val_loader = make_data_loader(
        val_dataset,
        loader_config,
        device=device,
        shuffle=False,
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
        mixed_precision=training_config.get("mixed_precision", True),
        channels_last=training_config.get("channels_last", True),
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

    print(f"Config copied to {config_copy_path}")
    print(f"Weights saved to {weights_path}")
    print(f"Training history saved to {history_path}")
    print(f"Loss curves saved to {curves_path}")


if __name__ == "__main__":
    main()
