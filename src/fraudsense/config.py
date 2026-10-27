"""
Configuration loader for FraudSense.
Reads and validates parameters from config/params.yaml.
"""

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: (str or Path) | None = None) -> dict[str, Any]:
    """
    Loads configuration dictionary from params.yaml.

    Args:
        config_path: Optional custom path to YAML config. Defaults to config/params.yaml.

    Returns:
        Dict containing hierarchical configuration settings.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "params.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
