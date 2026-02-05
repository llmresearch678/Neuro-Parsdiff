"""
Top-level training loop for Neuro-PARSDiff (Algorithm 4).
"""

import torch
from typing import Dict

from neuro_parsdiff.training.engine import train_one_epoch
from neuro_parsdiff.training.checkpoint import save_checkpoint


def train(
    model,
    block_predictor,
    dataloader,
    optimizer,
    schedule: Dict[str, torch.Tensor],
    num_node_states: int,
    num_edge_states: int,
    device: torch.device,
    num_epochs: int,
    lambda_ce: float,
    checkpoint_dir: str,
):
    """
    Train Neuro-PARSDiff.

    Parameters
    ----------
    model : nn.Module
    block_predictor : nn.Module
    dataloader : DataLoader
    optimizer : Optimizer
    schedule : dict
    num_node_states : int
    num_edge_states : int
    device : torch.device
    num_epochs : int
    lambda_ce : float
    checkpoint_dir : str
    """

    model.to(device)
    block_predictor.to(device)

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            block_predictor=block_predictor,
            dataloader=dataloader,
            optimizer=optimizer,
            schedule=schedule,
            num_node_states=num_node_states,
            num_edge_states=num_edge_states,
            device=device,
            lambda_ce=lambda_ce,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"Loss: {avg_loss:.6f}"
        )

        # Save checkpoint
        save_checkpoint(
            path=f"{checkpoint_dir}/epoch_{epoch}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
        )
