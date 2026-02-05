"""
Graph utilities for Neuro-PARSDiff.

Includes:
- Permutation-consistent node ranking (Algorithm 1)
- Block extraction Δ_k
- Mask construction M_k (Lemma 1)
- Graph structural features
"""

from .ranking import compute_node_ranking
from .blocks import get_block_indices
from .masking import build_block_mask
from .features import (
    compute_degree_features,
    compute_laplacian_features,
)

__all__ = [
    "compute_node_ranking",
    "get_block_indices",
    "build_block_mask",
    "compute_degree_features",
    "compute_laplacian_features",
]
