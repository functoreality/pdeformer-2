r"""This module provides a function to load a configuration file."""
from typing import Tuple
from omegaconf import OmegaConf, DictConfig


def load_config(file_path: str) -> DictConfig:
    r"""
    Load a configuration file.

    Args:
        file_path (str): The path of yaml configuration file.

    Returns:
        Tuple[dict, str]: The configuration dictionary and its string
            representation.
    """
    if not file_path.endswith(".yaml"):
        raise ValueError("The configuration file must be a yaml file")

    config = OmegaConf.load(file_path)

    base_config_path = config.get("base_config", "none")
    if base_config_path.lower() != "none":
        config_custom = config
        config = OmegaConf.load(base_config_path)
        config.merge_with(config_custom)

    return config
