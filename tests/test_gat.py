"""
Unit tests for GAT Encoder in PyTorch Geometric.
"""

import torch
from torch_geometric.data import Data

from fraudsense.models.gat import GATEncoder


def test_gat_encoder_forward_dimensions():
    """Verify 2-layer, 4-head GAT architecture produces 64-dim node embeddings."""
    encoder = GATEncoder(in_channels=4, hidden_channels=16, out_channels=64, heads=4)
    N = 10
    x = torch.randn((N, 4), dtype=torch.float)
    # Fully connected edge list for 10 nodes
    src = torch.repeat_interleave(torch.arange(N), N)
    dst = torch.tile(torch.arange(N), (N,))
    edge_index = torch.stack([src, dst], dim=0)

    out = encoder(x, edge_index)
    assert out.shape == (N, 64)
    assert not torch.isnan(out).any()


def test_gat_account_embedding_lookup():
    """Verify counterparty pair embedding lookup."""
    encoder = GATEncoder(in_channels=4, hidden_channels=16, out_channels=64, heads=4)
    N = 5
    x = torch.randn((N, 4), dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    pyg_data = Data(x=x, edge_index=edge_index, num_nodes=N)

    emb_single = encoder.get_account_embedding(pyg_data, payer_idx=1)
    assert emb_single.shape == (64,)

    emb_pair = encoder.get_account_embedding(pyg_data, payer_idx=1, payee_idx=2)
    assert emb_pair.shape == (64,)
