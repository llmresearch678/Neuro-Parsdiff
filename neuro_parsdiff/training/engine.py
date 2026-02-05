"""
Training engine for Neuro-PARSDiff.

Handles per-epoch training logic.
"""

import torch
from typing import Dict

from neuro_parsdiff.diffusion.forward import apply_block_noise
from neuro_parsdiff.diffusion.losses import hybrid_diffusion_loss
from neuro_parsdiff.graph.masking import build_block_mask


def train_one_epoch(
    model,
    block_predictor,
    dataloader,
    optimizer,
    schedule: Dict[str, torch.Tensor],
    num_node_states: int,
    num_edge_states: int,
    device: torch.device,
    lambda_ce: float,
) -> float:
    """
    Run one training epoch.

    Parameters
    ----------
    model : nn.Module
        Denoiser ℓ_φ.
    block_predictor : nn.Module
        Block predictor g_φ.
    dataloader : DataLoader
        Training data loader.
    optimizer : Optimizer
    schedule : dict
        Diffusion schedule.
    num_node_states : int
    num_edge_states : int
    device : torch.device
    lambda_ce : float

    Returns
    -------
    avg_loss : float
    """

    model.train()
    total_loss = 0.0
    num_batches = 0

    T = schedule["beta"].size(0)

    for batch in dataloader:
        # ---- unpack batch ----
        nodes = batch["nodes"].to(device)      # (N,)
        edges = batch["edges"].to(device)      # (N, N)
        num_nodes = nodes.size(0)

        # ---- sample diffusion step ----
        t = torch.randint(0, T, (1,)).item()
        beta_t = schedule["beta"][t]

        # ---- block prediction ----
        graph = {"nodes": nodes, "edges": edges}
        block_size = block_predictor(graph)
        block_indices = list(range(block_size))

        mask = build_block_mask(
            block_indices=block_indices,
            num_nodes=num_nodes,
            device=device,
        )

        # ---- forward diffusion ----
        noisy_graph = apply_block_noise(
            node_states=nodes,
            edge_states=edges,
            block_mask_nodes=mask["nodes"],
            block_mask_edges=mask["edges"],
            beta_t=beta_t,
            num_node_states=num_node_states,
            num_edge_states=num_edge_states,
        )

        # ---- denoising ----
        pred = model(
            noisy_graph=noisy_graph,
            timestep=t,
            mask=mask,
        )

        # ---- loss ----
        target = {"nodes": nodes, "edges": edges}
        loss = hybrid_diffusion_loss(
            pred=pred,
            target=target,
            mask=mask,
            lambda_ce=lambda_ce,
        )

        # ---- optimization ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)
