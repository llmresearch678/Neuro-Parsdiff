"""
I/O utilities for Neuro-PARSDiff.
"""

import os
import torch
from typing import Dict


def save_graph(
    graph: Dict[str, torch.Tensor],
    path: str,
):
    """
    Save a graph dictionary to disk.

    Parameters
    ----------
    graph : dict
        {"nodes": Tensor, "edges": Tensor}
    path : str
        Output file path (.pt)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(graph, path)


def load_graph(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Load a graph dictionary from disk.

    Parameters
    ----------
    path : str
        Input file path (.pt)
    device : torch.device

    Returns
    -------
    graph : dict
    """
    graph = torch.load(path, map_location=device)
    return graph
