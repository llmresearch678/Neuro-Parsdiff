"""
Minimum Spanning Tree (MST) based evaluation for Neuro-PARSDiff.

This module compares the MSTs of generated and ground-truth
connectomes to assess preservation of core structural backbones.
"""

import torch
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def _compute_mst(adj: torch.Tensor) -> np.ndarray:
    """
    Compute the MST of a graph given its adjacency matrix.

    Parameters
    ----------
    adj : Tensor (N, N)
        Weighted adjacency matrix.

    Returns
    -------
    mst : ndarray (N, N)
        Dense MST adjacency matrix.
    """
    adj_np = adj.detach().cpu().numpy()

    # Ensure non-negative weights
    adj_np = np.maximum(adj_np, 0.0)

    mst_sparse = minimum_spanning_tree(adj_np)
    mst = mst_sparse.toarray()

    return mst


def compute_mst_distance(
    pred_adj: torch.Tensor,
    gt_adj: torch.Tensor,
) -> float:
    """
    Compute distance between MSTs of predicted and ground-truth graphs.

    Uses mean squared error between MST adjacency matrices.

    Parameters
    ----------
    pred_adj : Tensor (N, N)
        Predicted adjacency matrix.
    gt_adj : Tensor (N, N)
        Ground-truth adjacency matrix.

    Returns
    -------
    dist : float
        MST distance score.
    """

    mst_pred = _compute_mst(pred_adj)
    mst_gt = _compute_mst(gt_adj)

    return float(np.mean((mst_pred - mst_gt) ** 2))
