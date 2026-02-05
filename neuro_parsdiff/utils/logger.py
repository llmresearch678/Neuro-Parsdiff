"""
Logging utilities for Neuro-PARSDiff.
"""

import logging
import sys


def get_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a standardized logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.

    Returns
    -------
    logger : logging.Logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
