"""
Reverse (denoising) diffusion process for Neuro-PARSDiff.

Implements p_φ(G_{t-1} | G_t) using a learned denoiser ℓ_φ.

This corresponds to:
- Algorithm 4 (training)
- Algorithm 5 (generation)

The denoiser predicts the clean graph G_0 distribution, from which
reverse transitions are analytically derived.
"""

from typing import Dict, Optional
import torch
import torch.nn.functional as F


@torch.no_grad()
def reverse_diffusion_step(
    denoiser,
    noisy_graph: Dict[str, torch.Tensor],
    timestep: int,
    mask: Dict[str, torch.Tensor],
    beta_t: float,
    sample: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Perform one reverse diffusion step: G_t → G_{t-1}.

    Parameters
    ----------
    denoiser : nn.Module
        Learned denoiser ℓ_φ. Must return logits for clean nodes & edges.
    noisy_graph : dict
        {
            "nodes": Tensor (N,),
            "edges": Tensor (N, N)
        }
    timestep : int
        Current diffusion timestep t.
    mask : dict
        {
            "nodes": Tensor (N,),
            "edges": Tensor (N, N)
        }
    beta_t : float
        Noise level at timestep t.
    sample : bool
        If True, sample from categorical distribution.
        If False, take argmax (deterministic).

    Returns
    -------
    prev_graph : dict
        {
            "nodes": Tensor (N,),
            "edges": Tensor (N, N)
        }
    """

    # --------------------------------------------------
    # 1. Predict clean graph distribution p_φ(G_0 | G_t)
    # --------------------------------------------------
    pred_logits = denoiser(
        noisy_graph=noisy_graph,
        timestep=timestep,
        mask=mask,
    )

    node_logits = pred_logits["nodes"]  # (N, Cn)
    edge_logits = pred_logits["edges"]  # (N, N, Ce)

    # --------------------------------------------------
    # 2. Convert logits → probabilities
    # --------------------------------------------------
    node_probs = F.softmax(node_logits, dim=-1)
    edge_probs = F.softmax(edge_logits, dim=-1)

    # --------------------------------------------------
    # 3. Sample or select clean states Ĝ_0
    # --------------------------------------------------
    if sample:
        clean_nodes = torch.multinomial(
            node_probs, num_samples=1
        ).squeeze(-1)

        N = clean_nodes.size(0)
        clean_edges = torch.multinomial(
            edge_probs.view(N * N, -1), num_samples=1
        ).view(N, N)

    else:
        clean_nodes = node_probs.argmax(dim=-1)
        clean_edges = edge_probs.argmax(dim=-1)

    # --------------------------------------------------
    # 4. Construct G_{t-1}
    #    Mix predicted clean graph with noisy input
    # --------------------------------------------------
    node_keep = torch.rand_like(clean_nodes.float()) > beta_t
    edge_keep = torch.rand_like(clean_edges.float()) > beta_t

    prev_nodes = torch.where(
        mask["nodes"] * node_keep,
        clean_nodes,
        noisy_graph["nodes"],
    )

    prev_edges = torch.where(
        mask["edges"] * edge_keep,
        clean_edges,
        noisy_graph["edges"],
    )

    return {
        "nodes": prev_nodes,
        "edges": prev_edges,
    }
