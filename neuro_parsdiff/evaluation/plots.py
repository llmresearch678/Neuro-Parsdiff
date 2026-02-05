"""
Visualization utilities for Neuro-PARSDiff evaluation.

This module provides plotting functions for:
- Adjacency matrix heatmaps
- Degree distribution comparison
- Laplacian spectrum visualization

Used for Figures 5–6 in the paper.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_adjacency_heatmap(
    adj: torch.Tensor,
    save_path: str,
    title: str = "Adjacency Matrix",
    cmap: str = "viridis",
):
    """
    Plot adjacency matrix as a heatmap.

    Parameters
    ----------
    adj : Tensor (N, N)
        Adjacency matrix.
    save_path : str
        Path to save the figure.
    title : str
        Plot title.
    cmap : str
        Colormap.
    """
    adj_np = adj.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(adj_np, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_degree_distribution(
    pred_adj: torch.Tensor,
    gt_adj: torch.Tensor,
    save_path: str,
    title: str = "Degree Distribution",
):
    """
    Plot degree distributions of predicted vs ground-truth graphs.

    Parameters
    ----------
    pred_adj : Tensor (N, N)
        Predicted adjacency.
    gt_adj : Tensor (N, N)
        Ground-truth adjacency.
    save_path : str
        Output file path.
    title : str
        Plot title.
    """

    pred_deg = pred_adj.sum(dim=1).detach().cpu().numpy()
    gt_deg = gt_adj.sum(dim=1).detach().cpu().numpy()

    plt.figure(figsize=(5, 4))
    plt.hist(
        gt_deg, bins=30, alpha=0.6,
        label="Ground Truth", density=True
    )
    plt.hist(
        pred_deg, bins=30, alpha=0.6,
        label="Generated", density=True
    )

    plt.xlabel("Degree")
    plt.ylabel("Density")
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_spectrum(
    adj: torch.Tensor,
    save_path: str,
    k: int = 20,
    title: str = "Laplacian Spectrum",
):
    """
    Plot the smallest k Laplacian eigenvalues.

    Parameters
    ----------
    adj : Tensor (N, N)
        Adjacency matrix.
    save_path : str
        Output file path.
    k : int
        Number of eigenvalues.
    title : str
        Plot title.
    """

    adj = adj.detach()
    deg = torch.diag(adj.sum(dim=1))
    lap = deg - adj

    eigvals = torch.linalg.eigvalsh(lap)
    eigvals = eigvals[:k].cpu().numpy()

    plt.figure(figsize=(5, 4))
    plt.plot(
        np.arange(1, k + 1),
        eigvals,
        marker="o"
    )
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
