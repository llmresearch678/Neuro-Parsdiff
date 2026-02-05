"""
Denoiser ℓ_φ for Neuro-PARSDiff (Algorithm 4).
"""

import torch
import torch.nn as nn

from .embeddings import GraphEmbeddings
from .transformer import GraphTransformer


class Denoiser(nn.Module):
    """
    Predicts clean graph logits p_φ(G_0 | G_t).
    """

    def __init__(
        self,
        num_node_states: int,
        num_edge_states: int,
        embed_dim: int,
        max_timesteps: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()

        self.embeddings = GraphEmbeddings(
            num_node_states,
            num_edge_states,
            embed_dim,
            max_timesteps,
        )

        self.transformer = GraphTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.node_head = nn.Linear(embed_dim, num_node_states)
        self.edge_head = nn.Linear(embed_dim, num_edge_states)

    def forward(
        self,
        noisy_graph: dict,
        timestep: int,
        mask: dict,
    ) -> dict:
        """
        Parameters
        ----------
        noisy_graph : dict
            {"nodes": (N,), "edges": (N, N)}
        timestep : int
        mask : dict
            {"nodes": (N,), "edges": (N, N)}

        Returns
        -------
        pred : dict
            {"nodes": (N, Cn), "edges": (N, N, Ce)}
        """

        nodes = noisy_graph["nodes"]
        edges = noisy_graph["edges"]

        node_emb, edge_emb, time_emb = self.embeddings(
            nodes, edges, timestep
        )

        # Inject timestep embedding
        node_emb = node_emb + time_emb

        # Transformer over nodes
        node_features = self.transformer(node_emb)

        # Predict node logits
        node_logits = self.node_head(node_features)

        # Predict edge logits (pairwise)
        N, D = node_features.size()
        edge_features = (
            node_features.unsqueeze(1) + node_features.unsqueeze(0)
        )  # (N, N, D)

        edge_logits = self.edge_head(edge_features)

        return {
            "nodes": node_logits,
            "edges": edge_logits,
        }
