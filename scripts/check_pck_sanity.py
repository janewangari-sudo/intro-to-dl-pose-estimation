"""Run a tiny synthetic sanity check for the reusable PCK implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pck import compute_pck, summarize_pck_counts


def main() -> None:
    """Verify correct, incorrect, and invisible keypoints are handled."""
    predicted = np.array([[0.0, 0.0], [8.0, 8.0], [5.0, 5.0]])
    target = np.array([[1.0, 1.0], [1.0, 1.0], [20.0, 20.0]])
    visibility = np.array([True, True, False])
    correct, visible = compute_pck(
        predicted,
        target,
        visibility,
        threshold=0.2,
        heatmap_size=(10, 10),
    )
    result = summarize_pck_counts(
        correct,
        visible,
        threshold=0.2,
        joint_names=["near", "far", "hidden"],
        method="synthetic",
    )

    assert result["mean_pck"] == 0.5
    assert result["per_joint_pck"]["near"] == 1.0
    assert result["per_joint_pck"]["far"] == 0.0
    assert result["per_joint_pck"]["hidden"] is None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
