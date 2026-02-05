"""
Block extraction utilities for Neuro-PARSDiff.

Splits ranked nodes into autoregressive blocks Δ_k.
"""

from typing import List
import torch


def get_block_indices(
    start: int,
    block_size: int,
    num_nodes: int,
) -> List[int]:
    """
    Get node indices for the current block Δ_k.

    Parameters
    ----------
    start : int
        Starting index in ranked list.
    block_size : int
        Size of the block.
    num_nodes : int
        Total number of nodes.

    Returns
    -------
    indices : list[int]
        Node indices in the block.
    """

    end = min(start + block_size, num_nodes)
    return list(range(start, end))
