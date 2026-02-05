"""
Evaluation module for Neuro-PARSDiff.

This package contains all evaluation routines used to assess
generated connectomes against ground-truth graphs.

Included evaluations:
- Quantitative metrics (MSE, Pearson, NMI, Spectral distance)
- Minimum Spanning Tree (MST) comparison
- Visualization utilities for plots and figures

These components support Tables 1 and Figures 4–6 in the paper.
"""

from .metrics import (
    compute_mse,
    compute_pearson,
    compute_nmi,
    compute_spectral_distance,
)

from .mst import (
    compute_mst_distance,
)

from .plots import (
    plot_adjacency_heatmap,
    plot_degree_distribution,
    plot_spectrum,
)

__all__ = [
    # Metrics
    "compute_mse",
    "compute_pearson",
    "compute_nmi",
    "compute_spectral_distance",

    # MST
    "compute_mst_distance",

    # Plots
    "plot_adjacency_heatmap",
    "plot_degree_distribution",
    "plot_spectrum",
]
