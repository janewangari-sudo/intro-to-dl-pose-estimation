"""Runtime device selection helpers."""

from __future__ import annotations

import torch


def select_device(device_name: str) -> torch.device:
    """Resolve ``cpu``, ``cuda``, or ``auto`` into a torch device."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)
