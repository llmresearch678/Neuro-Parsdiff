"""
Embedding layers for Neuro-PARSDiff.
"""

import torch
import torch.nn as nn


class GraphEmbeddings(nn.Module):
    """
    Node, edge, and timestep embeddings.
    """

    def __init__(
        self,
        num_node_states: int,
        num_edge_states: int,
        embed_dim: int,
        max_timesteps: int,
    ):
        super().__init__()

        self.node_embed = nn.Embedding(num_node_states, embed_dim)
        self.edge_embed = nn.Embedding(num_edge_states, embed_dim)
        self.time_embed = nn.Embedding(max_timesteps, embed_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        timestep: int,
    ):
        """
        Parameters
        ----------
        nodes : Tensor (N,)
        edges : Tensor (N, N)
        timestep : int

        Returns
        -------
        node_emb : Tensor (N, D)
        edge_emb : Tensor (N, N, D)
        time_emb : Tensor (D,)
        """
        node_emb = self.node_embed(nodes)
        edge_emb = self.edge_embed(edges)
        time_emb = self.time_embed(
            torch.tensor(timestep, device=nodes.device)
        )

        return node_emb, edge_emb, time_emb
