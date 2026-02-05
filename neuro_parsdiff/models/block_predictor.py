"""
Block-size predictor g_φ for Neuro-PARSDiff (Algorithm 2).
"""

import torch
import torch.nn as nn


class BlockPredictor(nn.Module):
    """
    Predicts the size of the next block Δ_k.
    """

    def __init__(
        self,
        embed_dim: int,
        max_block_size: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, max_block_size),
        )

    def forward(
        self,
        graph: dict,
    ) -> int:
        """
        Parameters
        ----------
        graph : dict
            {"nodes": (N,), "edges": (N, N)}

        Returns
        -------
        block_size : int
        """

        # Simple heuristic: average node state as embedding proxy
        node_states = graph["nodes"].float()
        graph_emb = node_states.mean().unsqueeze(0)

        logits = self.mlp(graph_emb)
        block_size = torch.argmax(logits, dim=-1).item() + 1

        return block_size
