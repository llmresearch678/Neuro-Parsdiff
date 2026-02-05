"""
Forward diffusion process for Neuro-PARSDiff.

This module implements the discrete-time forward noising kernel
q(G_t | G_{t-1}) with block-wise masking.

Only the current block Δ_i is corrupted, while the previously
constructed subgraph G_{≤ i-1} is kept clean and provides context.
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def forward_diffusion_step(
    node_states: torch.Tensor,
    edge_states: torch.Tensor,
    beta_t: float,
    num_node_states: int,
    num_edge_states: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply one step of discrete forward diffusion to node and edge states.

    Parameters
    ----------
    node_states : Tensor
        Shape (N,), integer categorical node states at time t-1.
    edge_states : Tensor
        Shape (N, N), integer categorical edge states at time t-1.
    beta_t : float
        Noise level at timestep t.
    num_node_states : int
        Number of possible node categories.
    num_edge_states : int
        Number of possible edge categories.

    Returns
    -------
    noisy_nodes : Tensor
        Shape (N,), node states at time t.
    noisy_edges : Tensor
        Shape (N, N), edge states at time t.
    """

    device = node_states.device
    N = node_states.size(0)

    # ---- Nodes ----
    # With probability beta_t, replace with random category
    node_noise = torch.rand(N, device=device) < beta_t
    random_nodes = torch.randint(
        low=0,
        high=num_node_states,
        size=(N,),
        device=device,
    )

    noisy_nodes = torch.where(node_noise, random_nodes, node_states)

    # ---- Edges ----
    edge_noise = torch.rand(N, N, device=device) < beta_t
    random_edges = torch.randint(
        low=0,
        high=num_edge_states,
        size=(N, N),
        device=device,
    )

    noisy_edges = torch.where(edge_noise, random_edges, edge_states)

    return noisy_nodes, noisy_edges


def apply_block_noise(
    node_states: torch.Tensor,
    edge_states: torch.Tensor,
    block_mask_nodes: torch.Tensor,
    block_mask_edges: torch.Tensor,
    beta_t: float,
    num_node_states: int,
    num_edge_states: int,
) -> Dict[str, torch.Tensor]:
    """
    Apply forward diffusion noise ONLY to the current block Δ_i.

    This corresponds to:
        G̃_t = M_i ⊙ q_t(G) + (1 - M_i) ⊙ G

    Parameters
    ----------
    node_states : Tensor
        Shape (N,), clean node states.
    edge_states : Tensor
        Shape (N, N), clean edge states.
    block_mask_nodes : Tensor
        Shape (N,), binary mask for nodes in Δ_i.
    block_mask_edges : Tensor
        Shape (N, N), binary mask for edges touching Δ_i.
    beta_t : float
        Diffusion noise level.
    num_node_states : int
        Number of node categories.
    num_edge_states : int
        Number of edge categories.

    Returns
    -------
    Dict with keys:
        - "nodes": noised node states
        - "edges": noised edge states
    """

    # First, fully noise everything (conceptually)
    noisy_nodes, noisy_edges = forward_diffusion_step(
        node_states=node_states,
        edge_states=edge_states,
        beta_t=beta_t,
        num_node_states=num_node_states,
        num_edge_states=num_edge_states,
    )

    # Apply block-wise masking
    final_nodes = (
        block_mask_nodes * noisy_nodes
        + (1 - block_mask_nodes) * node_states
    )

    final_edges = (
        block_mask_edges * noisy_edges
        + (1 - block_mask_edges) * edge_states
    )

    return {
        "nodes": final_nodes,
        "edges": final_edges,
    }
