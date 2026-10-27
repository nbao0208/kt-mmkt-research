"""
Configuration Utility for YAML Experiment Management.
Loads and recursively merges configuration files with support for parameter overrides.
"""

from typing import Dict, Any, Union
import os
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration dictionary from a YAML file.

    Args:
        config_path: Path to the .yaml configuration file.

    Returns:
        Dictionary representation of the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries where values in 'override' take precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary containing new or modified keys.

    Returns:
        Merged configuration dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
