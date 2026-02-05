"""
Graph generation via reverse diffusion for Neuro-PARSDiff.

Implements Algorithm 5 from the paper.
"""

import torch
from typing import Dict

from neuro_parsdiff.diffusion.reverse import reverse_diffusion_step
from neuro_parsdiff.graph.masking import build_block_mask
from neuro_parsdiff.graph.blocks import get_block_indices


@torch.no_grad()
def generate_graph(
    denoiser,
    block_predictor,
    num_nodes: int,
    schedule: Dict[str, torch.Tensor],
    num_node_states: int,
    num_edge_states: int,
    device: torch.device,
    sample: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generate a graph from scratch using Neuro-PARSDiff.

    Parameters
    ----------
    denoiser : nn.Module
        Trained denoiser ℓ_φ.
    block_predictor : nn.Module
        Block-size predictor g_φ.
    num_nodes : int
        Total number of nodes.
    schedule : dict
        Diffusion schedule (beta, alpha, alpha_bar).
    num_node_states : int
        Number of node categories.
    num_edge_states : int
        Number of edge categories.
    device : torch.device
        Torch device.
    sample : bool
        Whether to sample or take argmax.

    Returns
    -------
    graph : dict
        {
            "nodes": Tensor (N,),
            "edges": Tensor (N, N)
        }
    """

    T = schedule["beta"].size(0)

    # --- initialize with pure noise ---
    nodes = torch.randint(
        0, num_node_states, (num_nodes,), device=device
    )
    edges = torch.randint(
        0, num_edge_states, (num_nodes, num_nodes), device=device
    )

    graph = {"nodes": nodes, "edges": edges}

    # --- autoregressive block generation ---
    start = 0
    while start < num_nodes:
        block_size = block_predictor(graph)
        block_indices = get_block_indices(start, block_size, num_nodes)

        mask = build_block_mask(
            block_indices=block_indices,
            num_nodes=num_nodes,
            device=device,
        )

        # reverse diffusion for current block
        for t in reversed(range(T)):
            graph = reverse_diffusion_step(
                denoiser=denoiser,
                noisy_graph=graph,
                timestep=t,
                mask=mask,
                beta_t=schedule["beta"][t],
                sample=sample,
            )

        start += block_size

    return graph
