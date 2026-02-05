"""
Noise schedules for Neuro-PARSDiff.

Defines β_t, α_t, and cumulative ᾱ_t schedules used in
discrete-time graph diffusion.

These schedules control how aggressively node and edge
states are corrupted at each timestep.
"""

from typing import Dict
import torch
import math


def get_linear_noise_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Linear noise schedule.

    β_t increases linearly from beta_start to beta_end.

    Parameters
    ----------
    T : int
        Number of diffusion steps.
    beta_start : float
        Initial noise level.
    beta_end : float
        Final noise level.
    device : torch.device
        Torch device.

    Returns
    -------
    schedule : dict
        {
            "beta": Tensor (T,),
            "alpha": Tensor (T,),
            "alpha_bar": Tensor (T,)
        }
    """

    beta = torch.linspace(
        beta_start, beta_end, T, device=device
    )

    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
    }


def get_cosine_noise_schedule(
    T: int,
    s: float = 0.008,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Cosine noise schedule (improved stability).

    Based on:
        Nichol & Dhariwal, 2021

    ᾱ_t = cos^2( (t/T + s) / (1 + s) * π/2 )

    β_t is derived from ᾱ_t.

    Parameters
    ----------
    T : int
        Number of diffusion steps.
    s : float
        Small offset to prevent singularities.
    device : torch.device
        Torch device.

    Returns
    -------
    schedule : dict
        {
            "beta": Tensor (T,),
            "alpha": Tensor (T,),
            "alpha_bar": Tensor (T,)
        }
    """

    steps = torch.arange(
        T + 1, device=device, dtype=torch.float32
    )

    f = torch.cos(
        ((steps / T) + s) / (1 + s) * math.pi / 2
    ) ** 2

    alpha_bar = f[1:] / f[0]

    beta = 1.0 - (alpha_bar / torch.cat(
        [torch.ones(1, device=device), alpha_bar[:-1]]
    ))

    beta = torch.clamp(beta, min=1e-5, max=0.999)

    alpha = 1.0 - beta

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
    }


def get_noise_schedule(
    schedule_type: str,
    T: int,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Factory function for noise schedules.

    Parameters
    ----------
    schedule_type : str
        One of {"linear", "cosine"}.
    T : int
        Number of diffusion steps.
    device : torch.device
        Torch device.
    kwargs : dict
        Additional parameters for the schedule.

    Returns
    -------
    schedule : dict
        Noise schedule dictionary.
    """

    if schedule_type == "linear":
        return get_linear_noise_schedule(
            T=T, device=device, **kwargs
        )

    elif schedule_type == "cosine":
        return get_cosine_noise_schedule(
            T=T, device=device, **kwargs
        )

    else:
        raise ValueError(
            f"Unknown schedule_type: {schedule_type}"
        )
