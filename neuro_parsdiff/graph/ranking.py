"""
Permutation-consistent node ranking (Algorithm 1).

Produces an ordering ψ that is invariant to node permutations.
"""

import torch


def compute_node_ranking(
    adj: torch.Tensor,
) -> torch.Tensor:
    """
    Compute permutation-consistent node ranking ψ.

    Uses degree-based centrality (can be extended later).

    Parameters
    ----------
    adj : Tensor (N, N)
        Adjacency matrix.

    Returns
    -------
    ranking : Tensor (N,)
        Node indices sorted by importance (descending).
    """

    # Degree centrality
    degrees = adj.sum(dim=1)

    # Sort nodes by degree (descending)
    ranking = torch.argsort(degrees, descending=True)

    return ranking
