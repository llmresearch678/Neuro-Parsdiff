"""
Quantitative evaluation metrics for Neuro-PARSDiff.

Implements the core metrics used in Table 1:
- Mean Squared Error (MSE)
- Pearson Correlation
- Normalized Mutual Information (NMI)
- Spectral Distance (Laplacian-based)
"""

import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def compute_mse(
    pred_adj: torch.Tensor,
    gt_adj: torch.Tensor,
) -> float:
    """
    Mean Squared Error between adjacency matrices.

    Parameters
    ----------
    pred_adj : Tensor (N, N)
        Predicted adjacency matrix.
    gt_adj : Tensor (N, N)
        Ground-truth adjacency matrix.

    Returns
    -------
    mse : float
    """
    return torch.mean((pred_adj - gt_adj) ** 2).item()


def compute_pearson(
    pred_adj: torch.Tensor,
    gt_adj: torch.Tensor,
) -> float:
    """
    Pearson correlation between adjacency matrices.

    Parameters
    ----------
    pred_adj : Tensor (N, N)
    gt_adj : Tensor (N, N)

    Returns
    -------
    corr : float
    """
    pred = pred_adj.flatten()
    gt = gt_adj.flatten()

    pred = pred - pred.mean()
    gt = gt - gt.mean()

    numerator = torch.sum(pred * gt)
    denominator = torch.sqrt(
        torch.sum(pred ** 2) * torch.sum(gt ** 2)
    ) + 1e-8

    return (numerator / denominator).item()


def compute_nmi(
    pred_blocks: np.ndarray,
    gt_blocks: np.ndarray,
) -> float:
    """
    Normalized Mutual Information (NMI) between block assignments.

    Used to evaluate whether generated blocks preserve
    ground-truth community structure.

    Parameters
    ----------
    pred_blocks : ndarray (N,)
        Predicted block labels.
    gt_blocks : ndarray (N,)
        Ground-truth block labels.

    Returns
    -------
    nmi : float
    """
    return normalized_mutual_info_score(
        gt_blocks, pred_blocks
    )


def compute_spectral_distance(
    pred_adj: torch.Tensor,
    gt_adj: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Spectral distance based on Laplacian eigenvalues.

    Measures global structural similarity between graphs.

    Parameters
    ----------
    pred_adj : Tensor (N, N)
    gt_adj : Tensor (N, N)
    k : int
        Number of smallest eigenvalues to compare.

    Returns
    -------
    dist : float
    """

    def laplacian_eigs(adj: torch.Tensor, k: int):
        deg = torch.diag(adj.sum(dim=1))
        lap = deg - adj
        eigvals = torch.linalg.eigvalsh(lap)
        return eigvals[:k]

    eig_pred = laplacian_eigs(pred_adj, k)
    eig_gt = laplacian_eigs(gt_adj, k)

    return torch.mean((eig_pred - eig_gt) ** 2).item()
