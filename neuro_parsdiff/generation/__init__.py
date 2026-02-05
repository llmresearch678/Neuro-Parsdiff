"""
Graph generation module for Neuro-PARSDiff.

Implements:
- Full autoregressive graph generation (Algorithm 5)
- Partial / conditional graph reconstruction
"""

from .sample import generate_graph
from .reconstruct import reconstruct_graph

__all__ = [
    "generate_graph",
    "reconstruct_graph",
]
