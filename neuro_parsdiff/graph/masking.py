"""
Mask construction for Neuro-PARSDiff.

Implements block-wise masking M_k (Lemma 1).
"""

from typing import Dict, List
import torch


def build_block_mask(
    block_indices: List[int],
    num_nodes: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Construct node and edge masks for a block Δ_k.

    Parameters
    ----------
    block_indices : list[int]
        Indices of nodes in the current block.
    num_nodes : int
        Total number of nodes.
    device : torch.device

    Returns
    -------
    mask : dict
        {
            "nodes": Tensor (N,),
            "edges": Tensor (N, N)
        }
    """

    node_mask = torch.zeros(num_nodes, device=device)
    node_mask[block_indices] = 1.0

    edge_mask = torch.zeros(num_nodes, num_nodes, device=device)

    for i in block_indices:
        edge_mask[i, :] = 1.0
        edge_mask[:, i] = 1.0

    return {
        "nodes": node_mask,
        "edges": edge_mask,
    }
