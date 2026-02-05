"""
Checkpointing utilities for Neuro-PARSDiff.
"""

import torch
from typing import Optional


def save_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
):
    """
    Save training checkpoint.

    Parameters
    ----------
    path : str
    model : nn.Module
    optimizer : Optimizer
    epoch : int
    """

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model,
    optimizer: Optional[object] = None,
):
    """
    Load training checkpoint.

    Parameters
    ----------
    path : str
    model : nn.Module
    optimizer : Optimizer (optional)

    Returns
    -------
    epoch : int
    """

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["epoch"]
