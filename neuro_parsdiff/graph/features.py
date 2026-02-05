"""
Graph feature extraction for Neuro-PARSDiff.

Computes degree- and Laplacian-based features.
"""

import torch


def compute_degree_features(
    adj: torch.Tensor,
) -> torch.Tensor:
    """
    Compute node degree features.

    Parameters
    ----------
    adj : Tensor (N, N)

    Returns
    -------
    degree : Tensor (N, 1)
    """
    degree = adj.sum(dim=1, keepdim=True)
    return degree


def compute_laplacian_features(
    adj: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Compute Laplacian eigenvalue features.

    Parameters
    ----------
    adj : Tensor (N, N)
    k : int
        Number of eigenvalues.

    Returns
    -------
    features : Tensor (k,)
    """

    deg = torch.diag(adj.sum(dim=1))
    lap = deg - adj

    eigvals = torch.linalg.eigvalsh(lap)
    return eigvals[:k]
