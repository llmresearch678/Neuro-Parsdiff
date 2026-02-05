"""
Neuro-PARSDiff Diffusion Module
==============================

This subpackage implements the discrete-time block-wise diffusion
process used in Neuro-PARSDiff.

It includes:
- Forward noising kernels q(G_t | G_{t-1})
- Reverse denoising transitions p_φ(G_{t-1} | G_t)
- Noise schedules (β_t, α_t)
- Diffusion and reconstruction losses

The implementation follows:
Algorithm 4 (Block-wise Diffusion Denoising)
Algorithm 5 (Autoregressive Graph Generation)

All diffusion operations are permutation-invariant and
block-mask aware.
"""

from .schedules import (
    get_linear_noise_schedule,
    get_cosine_noise_schedule,
)

from .forward import (
    forward_diffusion_step,
    apply_block_noise,
)

from .reverse import (
    reverse_diffusion_step,
)

from .losses import (
    diffusion_loss,
    reconstruction_loss,
    hybrid_diffusion_loss,
)

__all__ = [
    # Schedules
    "get_linear_noise_schedule",
    "get_cosine_noise_schedule",

    # Forward process
    "forward_diffusion_step",
    "apply_block_noise",

    # Reverse process
    "reverse_diffusion_step",

    # Losses
    "diffusion_loss",
    "reconstruction_loss",
    "hybrid_diffusion_loss",
]
