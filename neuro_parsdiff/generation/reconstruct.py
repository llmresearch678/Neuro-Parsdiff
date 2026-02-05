"""
Partial graph reconstruction for Neuro-PARSDiff.

Used for conditional generation and inpainting.
"""

import torch
from typing import Dict

from neuro_parsdiff.diffusion.reverse import reverse_diffusion_step


@torch.no_grad()
def reconstruct_graph(
    denoiser,
    noisy_graph: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
    schedule: Dict[str, torch.Tensor],
    sample: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct a partially observed graph.

    Parameters
    ----------
    denoiser : nn.Module
        Trained denoiser ℓ_φ.
    noisy_graph : dict
        Observed + noised graph.
    mask : dict
        Binary mask indicating missing parts.
    schedule : dict
        Diffusion schedule.
    sample : bool
        Sample or argmax.

    Returns
    -------
    graph : dict
        Reconstructed graph.
    """

    graph = noisy_graph
    T = schedule["beta"].size(0)

    for t in reversed(range(T)):
        graph = reverse_diffusion_step(
            denoiser=denoiser,
            noisy_graph=graph,
            timestep=t,
            mask=mask,
            beta_t=schedule["beta"][t],
            sample=sample,
        )

    return graph
