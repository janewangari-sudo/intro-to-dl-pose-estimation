"""COCO person-keypoint dataset and heatmap target generation."""

from __future__ import annotations

import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .transforms import crop_resize_normalize


COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def download_and_unzip(url: str, target_dir: str | Path) -> None:
    """Download a ZIP archive if needed and extract it into ``target_dir``."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / url.rsplit("/", maxsplit=1)[-1]

    if not archive_path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, archive_path)

    print(f"Extracting {archive_path.name}")
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(target_dir)


def prepare_coco_val2017(
    data_dir: str | Path,
    download: bool = True,
) -> Tuple[Path, Path]:
    """Ensure COCO val2017 images and person-keypoint annotations exist."""
    data_dir = Path(data_dir)
    image_dir = data_dir / "val2017"
    annotation_file = (
        data_dir / "annotations" / "person_keypoints_val2017.json"
    )

    if download:
        if not image_dir.exists():
            download_and_unzip(COCO_VAL_IMAGES_URL, data_dir)
        if not annotation_file.exists():
            download_and_unzip(COCO_ANNOTATIONS_URL, data_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"COCO image directory not found: {image_dir}. "
            "Enable data.download or place val2017 there."
        )
    if not annotation_file.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: {annotation_file}. "
            "Enable data.download or place the annotations there."
        )

    return image_dir, annotation_file


def get_person_image_ids(
    coco: COCO,
    min_keypoints: int = 5,
) -> list[int]:
    """Return image IDs containing a non-crowd person with enough keypoints."""
    person_category_id = coco.getCatIds(catNms=["person"])[0]
    annotation_ids = coco.getAnnIds(
        catIds=person_category_id,
        iscrowd=False,
    )
    annotations = coco.loadAnns(annotation_ids)

    # The list(set(...)) form matches the image selection in the notebook.
    return list(
        {
            annotation["image_id"]
            for annotation in annotations
            if annotation["num_keypoints"] >= min_keypoints
        }
    )


def split_image_ids(
    image_ids: Iterable[int],
    train_fraction: float = 0.85,
    seed: int = 42,
) -> Tuple[list[int], list[int]]:
    """Shuffle image IDs deterministically and split them into train/val sets."""
    shuffled_ids = list(image_ids)
    random.Random(seed).shuffle(shuffled_ids)
    split_index = int(train_fraction * len(shuffled_ids))
    return shuffled_ids[:split_index], shuffled_ids[split_index:]


class COCOPoseDataset(Dataset):
    """One COCO person crop and its 17 Gaussian keypoint heatmaps per sample."""

    def __init__(
        self,
        coco_api: COCO,
        image_dir: str | Path,
        image_ids: Iterable[int],
        input_size: Sequence[int] = (256, 192),
        heatmap_size: Sequence[int] = (64, 48),
        sigma: float = 2.0,
        min_keypoints: int = 1,
    ) -> None:
        self.coco = coco_api
        self.image_dir = Path(image_dir)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.heatmap_size = (
            int(heatmap_size[0]),
            int(heatmap_size[1]),
        )
        self.sigma = float(sigma)
        self.num_joints = 17
        self.person_category_id = coco_api.getCatIds(catNms=["person"])[0]
        self.samples = []

        for image_id in image_ids:
            annotation_ids = coco_api.getAnnIds(
                imgIds=image_id,
                catIds=self.person_category_id,
                iscrowd=False,
            )
            for annotation in coco_api.loadAnns(annotation_ids):
                if annotation["num_keypoints"] >= min_keypoints:
                    self.samples.append((image_id, annotation))

    def __len__(self) -> int:
        return len(self.samples)

    def _make_heatmaps(
        self,
        keypoints: np.ndarray,
        bbox: Sequence[float],
    ) -> torch.Tensor:
        """Create the same integer-centered Gaussian heatmaps as the notebook."""
        heatmap_height, heatmap_width = self.heatmap_size
        heatmaps = np.zeros(
            (self.num_joints, heatmap_height, heatmap_width),
            dtype=np.float32,
        )
        box_x, box_y, box_width, box_height = bbox
        if box_width <= 0 or box_height <= 0:
            return torch.from_numpy(heatmaps)

        grid_x, grid_y = np.meshgrid(
            np.arange(heatmap_width),
            np.arange(heatmap_height),
        )
        for joint_index, (keypoint_x, keypoint_y, visibility) in enumerate(
            keypoints
        ):
            if visibility == 0:
                continue

            heatmap_x = (keypoint_x - box_x) / box_width * heatmap_width
            heatmap_y = (keypoint_y - box_y) / box_height * heatmap_height
            center_x = int(heatmap_x)
            center_y = int(heatmap_y)

            if (
                0 <= center_x < heatmap_width
                and 0 <= center_y < heatmap_height
            ):
                squared_distance = (
                    (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
                )
                heatmaps[joint_index] = np.exp(
                    -squared_distance / (2 * self.sigma**2)
                )

        return torch.from_numpy(heatmaps)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id, annotation = self.samples[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        image_tensor = crop_resize_normalize(
            image,
            annotation["bbox"],
            self.input_size,
        )
        keypoints = (
            np.asarray(annotation["keypoints"])
            .reshape(-1, 3)
            .astype(np.float32)
        )
        heatmaps = self._make_heatmaps(keypoints, annotation["bbox"])
        return image_tensor, heatmaps


def build_coco_datasets(
    data_dir: str | Path,
    input_size: Sequence[int] = (256, 192),
    heatmap_size: Sequence[int] = (64, 48),
    sigma: float = 2.0,
    train_fraction: float = 0.85,
    split_seed: int = 42,
    min_image_keypoints: int = 5,
    min_annotation_keypoints: int = 1,
    download: bool = True,
) -> Tuple[COCOPoseDataset, COCOPoseDataset]:
    """Prepare COCO and construct the notebook-equivalent train/val datasets."""
    image_dir, annotation_file = prepare_coco_val2017(data_dir, download)
    coco = COCO(str(annotation_file))
    image_ids = get_person_image_ids(coco, min_image_keypoints)
    train_ids, val_ids = split_image_ids(
        image_ids,
        train_fraction=train_fraction,
        seed=split_seed,
    )

    dataset_options = {
        "coco_api": coco,
        "image_dir": image_dir,
        "input_size": input_size,
        "heatmap_size": heatmap_size,
        "sigma": sigma,
        "min_keypoints": min_annotation_keypoints,
    }
    train_dataset = COCOPoseDataset(image_ids=train_ids, **dataset_options)
    val_dataset = COCOPoseDataset(image_ids=val_ids, **dataset_options)
    return train_dataset, val_dataset
