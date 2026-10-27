"""
Graph Attention Network (GAT) encoder for FraudSense.
2 layers, 4 heads, embedding dimension 64.
Computes inductive account embeddings using neighbor sampling and PyG Data.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv

from ..config import load_config


class GATEncoder(nn.Module):
    """
    2-layer Graph Attention Network with multi-head attention.

    Architecture:
    - Layer 1: GATConv(in_dim -> 16, heads=4) => 64 dimensions, ELU activation
    - Layer 2: GATConv(64 -> 16, heads=4) => 64 dimensions (or 64 single head)
    Outputs 64-dimensional node representation.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 16,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        cfg = config or load_config()
        gat_cfg = cfg.get("models", {}).get("gat", {})
        self.embedding_dim: int = int(gat_cfg.get("embedding_dim", out_channels))
        self.heads: int = int(gat_cfg.get("heads", heads))
        self.dropout = dropout

        # Layer 1: in_channels -> hidden_channels * heads (16 * 4 = 64)
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=self.heads,
            dropout=self.dropout,
        )

        # Layer 2: 64 -> embedding_dim (heads concatenated or averaged)
        # 16 * 4 = 64
        self.conv2 = GATConv(
            in_channels=hidden_channels * self.heads,
            out_channels=hidden_channels,
            heads=self.heads,
            dropout=self.dropout,
        )

        self.out_proj = nn.Linear(hidden_channels * self.heads, self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass computing node embeddings.

        Args:
            x: Node features tensor of shape (N, in_channels).
            edge_index: Edge index tensor of shape (2, E).
            edge_weight: Optional edge weight tensor of shape (E,).

        Returns:
            Tensor of node embeddings of shape (N, 64).
        """
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.elu(h)

        out = self.out_proj(h)
        return out

    def get_account_embedding(
        self,
        pyg_data: Data,
        payer_idx: int,
        payee_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Extracts embedding for an individual transaction counterparty pair.
        If payee_idx is provided, returns averaged embedding of payer and payee;
        otherwise returns payer embedding.

        Returns:
            Tensor of shape (embedding_dim,).
        """
        self.eval()
        with torch.no_grad():
            node_embeds = self.forward(
                pyg_data.x,
                pyg_data.edge_index,
                pyg_data.edge_weight,
            )
            if payer_idx >= len(node_embeds):
                return torch.zeros(self.embedding_dim, dtype=torch.float)

            payer_emb = node_embeds[payer_idx]
            if payee_idx is not None and payee_idx < len(node_embeds):
                payee_emb = node_embeds[payee_idx]
                return (payer_emb + payee_emb) / 2.0
            return payer_emb

    def fit_unsupervised(
        self,
        pyg_data: Data,
        epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 512,
    ) -> "GATEncoder":
        """
        Trains GAT encoder using link reconstruction / neighbor sampling.
        Ensures embeddings encode topological graph proximity.
        """
        if pyg_data.num_nodes == 0 or pyg_data.edge_index.shape[1] == 0:
            return self

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        loader = NeighborLoader(
            pyg_data,
            num_neighbors=[10, 5],
            batch_size=batch_size,
            input_nodes=None,
            shuffle=True,
        )

        self.train()
        for _ in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                z = self.forward(batch.x, batch.edge_index)
                # Link reconstruction loss on batch edges
                if batch.edge_index.shape[1] > 0:
                    src, dst = batch.edge_index[0], batch.edge_index[1]
                    # Valid indices within batch
                    valid_mask = (src < len(z)) & (dst < len(z))
                    if valid_mask.sum() > 0:
                        s_idx = src[valid_mask]
                        d_idx = dst[valid_mask]
                        pos_scores = (z[s_idx] * z[d_idx]).sum(dim=1)
                        # Negative sampling
                        neg_dst = torch.randint(0, len(z), (len(s_idx),), device=z.device)
                        neg_scores = (z[s_idx] * z[neg_dst]).sum(dim=1)

                        loss = (
                            -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
                            - torch.log(1.0 - torch.sigmoid(neg_scores) + 1e-8).mean()
                        )
                        loss.backward()
                        optimizer.step()

        self.eval()
        return self
