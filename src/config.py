"""Configuration and path helpers shared by the command-line scripts."""

from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def project_path(path_value: str | Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_path(path_value: str | Path) -> Path:
    """Resolve and validate a path that must remain under ``outputs/``."""
    path = project_path(path_value).resolve()
    outputs_root = OUTPUTS_ROOT.resolve()
    if path != outputs_root and outputs_root not in path.parents:
        raise ValueError(f"Output path must be under {outputs_root}: {path}")
    return path
