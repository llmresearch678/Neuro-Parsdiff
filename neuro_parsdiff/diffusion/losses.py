"""
Loss functions for Neuro-PARSDiff.

Implements the hybrid diffusion objective combining:
- Diffusion consistency loss
- Categorical reconstruction (cross-entropy) loss

This corresponds to Eq. (3) and Algorithm 4 in the paper.
"""

from typing import Dict
import torch
import torch.nn.functional as F


def diffusion_loss(
    pred_nodes: torch.Tensor,
    pred_edges: torch.Tensor,
    target_nodes: torch.Tensor,
    target_edges: torch.Tensor,
    mask_nodes: torch.Tensor,
    mask_edges: torch.Tensor,
) -> torch.Tensor:
    """
    Diffusion reconstruction loss.

    Measures how well the denoiser reconstructs the clean
    graph G_0 from a partially noised graph G_t.

    Uses masked mean-squared error (can be swapped later).

    Parameters
    ----------
    pred_nodes : Tensor
        Shape (N, Cn), predicted logits or embeddings for nodes.
    pred_edges : Tensor
        Shape (N, N, Ce), predicted logits or embeddings for edges.
    target_nodes : Tensor
        Shape (N,), ground-truth node categories.
    target_edges : Tensor
        Shape (N, N), ground-truth edge categories.
    mask_nodes : Tensor
        Shape (N,), binary mask for current block.
    mask_edges : Tensor
        Shape (N, N), binary mask for current block edges.

    Returns
    -------
    loss : Tensor
        Scalar diffusion loss.
    """

    # Convert targets to one-hot
    target_nodes_oh = F.one_hot(
        target_nodes, num_classes=pred_nodes.size(-1)
    ).float()

    target_edges_oh = F.one_hot(
        target_edges, num_classes=pred_edges.size(-1)
    ).float()

    # Node diffusion loss
    node_loss = F.mse_loss(
        pred_nodes * mask_nodes.unsqueeze(-1),
        target_nodes_oh * mask_nodes.unsqueeze(-1),
        reduction="sum",
    )

    # Edge diffusion loss
    edge_loss = F.mse_loss(
        pred_edges * mask_edges.unsqueeze(-1),
        target_edges_oh * mask_edges.unsqueeze(-1),
        reduction="sum",
    )

    # Normalize by number of active elements
    node_norm = mask_nodes.sum().clamp(min=1.0)
    edge_norm = mask_edges.sum().clamp(min=1.0)

    loss = node_loss / node_norm + edge_loss / edge_norm
    return loss


def reconstruction_loss(
    pred_nodes: torch.Tensor,
    pred_edges: torch.Tensor,
    target_nodes: torch.Tensor,
    target_edges: torch.Tensor,
    mask_nodes: torch.Tensor,
    mask_edges: torch.Tensor,
) -> torch.Tensor:
    """
    Categorical reconstruction loss (cross-entropy).

    Directly maximizes p_φ(G_0 | G_t) for nodes and edges,
    restricted to the current block.

    Parameters
    ----------
    pred_nodes : Tensor
        Shape (N, Cn), node logits.
    pred_edges : Tensor
        Shape (N, N, Ce), edge logits.
    target_nodes : Tensor
        Shape (N,), true node categories.
    target_edges : Tensor
        Shape (N, N), true edge categories.
    mask_nodes : Tensor
        Shape (N,), binary node mask.
    mask_edges : Tensor
        Shape (N, N), binary edge mask.

    Returns
    -------
    loss : Tensor
        Scalar cross-entropy loss.
    """

    # Node CE
    node_ce = F.cross_entropy(
        pred_nodes,
        target_nodes,
        reduction="none",
    )

    node_ce = (node_ce * mask_nodes).sum() / mask_nodes.sum().clamp(min=1.0)

    # Edge CE (flatten for efficiency)
    N = target_nodes.size(0)
    edge_ce = F.cross_entropy(
        pred_edges.view(N * N, -1),
        target_edges.view(-1),
        reduction="none",
    )

    edge_ce = (
        edge_ce * mask_edges.view(-1)
    ).sum() / mask_edges.sum().clamp(min=1.0)

    return node_ce + edge_ce


def hybrid_diffusion_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
    lambda_ce: float = 0.1,
) -> torch.Tensor:
    """
    Hybrid diffusion objective used for training.

    L_t = L_diff + λ * L_CE

    Parameters
    ----------
    pred : dict
        {
            "nodes": Tensor (N, Cn),
            "edges": Tensor (N, N, Ce)
        }
    target : dict
        {
            "nodes": Tensor (N,),
            "edges": Tensor (N, N)
        }
    mask : dict
        {
            "nodes": Tensor (N,),
            "edges": Tensor (N, N)
        }
    lambda_ce : float
        Weight of reconstruction loss.

    Returns
    -------
    loss : Tensor
        Scalar hybrid loss.
    """

    l_diff = diffusion_loss(
        pred_nodes=pred["nodes"],
        pred_edges=pred["edges"],
        target_nodes=target["nodes"],
        target_edges=target["edges"],
        mask_nodes=mask["nodes"],
        mask_edges=mask["edges"],
    )

    l_ce = reconstruction_loss(
        pred_nodes=pred["nodes"],
        pred_edges=pred["edges"],
        target_nodes=target["nodes"],
        target_edges=target["edges"],
        mask_nodes=mask["nodes"],
        mask_edges=mask["edges"],
    )

    return l_diff + lambda_ce * l_ce
