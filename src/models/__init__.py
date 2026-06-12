"""Pose estimation model definitions."""

from .simplebaseline import SimpleBaselinePoseNet
from .trivial_baseline import AveragePoseTemplate, estimate_average_pose

__all__ = [
    "AveragePoseTemplate",
    "SimpleBaselinePoseNet",
    "estimate_average_pose",
]
