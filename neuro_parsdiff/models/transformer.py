"""
Graph Transformer backbone for Neuro-PARSDiff.
"""

import torch
import torch.nn as nn


class GraphTransformer(nn.Module):
    """
    Transformer operating on node embeddings with global attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        node_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_emb : Tensor (N, D)

        Returns
        -------
        out : Tensor (N, D)
        """
        # Add batch dimension
        x = node_emb.unsqueeze(0)  # (1, N, D)
        x = self.encoder(x)
        return x.squeeze(0)
