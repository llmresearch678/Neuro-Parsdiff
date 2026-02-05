"""
Configuration loading utilities for Neuro-PARSDiff.
"""

import yaml
from typing import Dict


def load_config(
    path: str,
) -> Dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to YAML config.

    Returns
    -------
    config : dict
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
