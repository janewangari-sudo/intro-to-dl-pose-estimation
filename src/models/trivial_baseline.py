"""Non-neural average-pose baseline for COCO keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AveragePoseTemplate:
    """Fixed bbox-normalized pose estimated from training annotations."""

    normalized_coordinates: np.ndarray
    sample_counts: np.ndarray

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.normalized_coordinates, dtype=np.float64)
        counts = np.asarray(self.sample_counts, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                "normalized_coordinates must have shape (num_joints, 2)"
            )
        if counts.shape != (coordinates.shape[0],):
            raise ValueError(
                "sample_counts must contain one value for each joint"
            )

        object.__setattr__(self, "normalized_coordinates", coordinates)
        object.__setattr__(self, "sample_counts", counts)

    @property
    def valid_joints(self) -> np.ndarray:
        """Return a mask for joints observed at least once during fitting."""
        return self.sample_counts > 0

    def to_heatmap_coordinates(
        self,
        heatmap_size: Sequence[int],
    ) -> np.ndarray:
        """Convert normalized template points to ``(x, y)`` heatmap points."""
        heatmap_height, heatmap_width = (
            int(heatmap_size[0]),
            int(heatmap_size[1]),
        )
        coordinates = self.normalized_coordinates.copy()
        coordinates[:, 0] *= heatmap_width
        coordinates[:, 1] *= heatmap_height
        coordinates[~self.valid_joints] = np.nan
        return coordinates

    def to_dict(self, joint_names: Sequence[str]) -> dict:
        """Return a JSON-serializable representation of the template."""
        if len(joint_names) != len(self.sample_counts):
            raise ValueError("joint_names must match the template joint count")

        per_joint = {}
        for joint_name, coordinates, count in zip(
            joint_names,
            self.normalized_coordinates,
            self.sample_counts,
        ):
            is_valid = int(count) > 0
            per_joint[joint_name] = {
                "x_norm": float(coordinates[0]) if is_valid else None,
                "y_norm": float(coordinates[1]) if is_valid else None,
                "sample_count": int(count),
                "valid": is_valid,
            }

        return {
            "method": "trivial_average_pose",
            "coordinate_system": "normalized_person_bbox",
            "per_joint_template": per_joint,
        }


def estimate_average_pose(
    samples: Iterable[tuple[int, Mapping]],
    num_joints: int = 17,
) -> AveragePoseTemplate:
    """Estimate mean normalized joint locations from COCO annotations.

    Only keypoints with visibility ``v > 0`` contribute. Annotations with
    non-positive or non-finite bbox dimensions are ignored. Missing or
    malformed keypoint entries are skipped without affecting other samples.
    """
    coordinate_sums = np.zeros((num_joints, 2), dtype=np.float64)
    sample_counts = np.zeros(num_joints, dtype=np.int64)

    for _, annotation in samples:
        bbox = np.asarray(annotation.get("bbox", []), dtype=np.float64)
        if bbox.size < 4:
            continue
        bbox_x, bbox_y, bbox_width, bbox_height = bbox[:4]
        if (
            not np.all(np.isfinite(bbox[:4]))
            or bbox_width <= 0
            or bbox_height <= 0
        ):
            continue

        flat_keypoints = np.asarray(
            annotation.get("keypoints", []),
            dtype=np.float64,
        )
        available_joints = min(flat_keypoints.size // 3, num_joints)
        if available_joints == 0:
            continue

        keypoints = flat_keypoints[: available_joints * 3].reshape(-1, 3)
        for joint_index, (x_coord, y_coord, visibility) in enumerate(
            keypoints
        ):
            if (
                not np.isfinite(visibility)
                or visibility <= 0
                or not np.isfinite(x_coord)
                or not np.isfinite(y_coord)
            ):
                continue

            coordinate_sums[joint_index, 0] += (
                x_coord - bbox_x
            ) / bbox_width
            coordinate_sums[joint_index, 1] += (
                y_coord - bbox_y
            ) / bbox_height
            sample_counts[joint_index] += 1

    normalized_coordinates = np.full(
        (num_joints, 2),
        np.nan,
        dtype=np.float64,
    )
    valid_joints = sample_counts > 0
    normalized_coordinates[valid_joints] = (
        coordinate_sums[valid_joints] / sample_counts[valid_joints, None]
    )
    return AveragePoseTemplate(normalized_coordinates, sample_counts)
