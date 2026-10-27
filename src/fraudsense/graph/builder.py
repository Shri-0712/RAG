"""
Dynamic transaction graph builder using PyTorch Geometric.
Constructs account-transaction graphs with temporal recency-weighted edge decay.
"""

from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected

from ..config import load_config
from ..ingest.schema import TransactionEvent


class DynamicGraphBuilder:
    """
    Maintains a dynamic, time-decayed transaction network.

    Nodes represent accounts/counterparties (payers and payees).
    Edges represent payment transactions with recency-weighted exponential decay.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        gat_cfg = cfg.get("models", {}).get("gat", {})
        self.recency_window_hours: float = float(gat_cfg.get("recency_window_hours", 72.0))
        self.decay_tau_seconds: float = self.recency_window_hours * 3600.0

        # Mapping: entity_id -> node_index
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}

        # Node stats: [out_degree, in_degree, total_amount_sent, total_amount_recv]
        self._node_stats: list[list[float]] = []

        # Edges: (src_idx, dst_idx, amount, timestamp)
        self.edges: list[tuple[int, int, float, datetime]] = []

    def _get_or_create_node(self, entity_id: str) -> int:
        """Retrieves node index or registers new entity node."""
        if entity_id in self.node_to_idx:
            return self.node_to_idx[entity_id]

        idx = len(self.node_to_idx)
        self.node_to_idx[entity_id] = idx
        self.idx_to_node[idx] = entity_id
        # Initialize node stats: [out_degree, in_degree, total_amount_sent, total_amount_recv]
        self._node_stats.append([0.0, 0.0, 0.0, 0.0])
        return idx

    def add_event(self, event: TransactionEvent) -> None:
        """
        Ingests a payment event into the dynamic graph.

        Args:
            event: Normalized TransactionEvent instance.
        """
        src_idx = self._get_or_create_node(event.payer_id)
        dst_idx = self._get_or_create_node(event.payee_id)

        # Update node statistics
        self._node_stats[src_idx][0] += 1.0  # out_degree
        self._node_stats[src_idx][2] += event.amount  # total_sent
        self._node_stats[dst_idx][1] += 1.0  # in_degree
        self._node_stats[dst_idx][3] += event.amount  # total_recv

        # Record edge
        self.edges.append((src_idx, dst_idx, event.amount, event.ts))

    def build_pyg_data(self, current_time: datetime | None = None) -> Data:
        """
        Constructs a PyTorch Geometric Data object with recency-weighted edge weights.

        Args:
            current_time: Reference timestamp for edge decay calculation.
                          If None, uses the latest edge timestamp.

        Returns:
            torch_geometric.data.Data object with x, edge_index, edge_attr, and edge_weight.
        """
        num_nodes = len(self.node_to_idx)
        if num_nodes == 0:
            return Data(
                x=torch.zeros((0, 4), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2), dtype=torch.float),
                edge_weight=torch.zeros((0,), dtype=torch.float),
            )

        # Construct node feature matrix (normalized stats)
        stats = np.array(self._node_stats, dtype=np.float32)
        # Log-transform amount fields for numerical stability
        stats[:, 2] = np.log1p(np.maximum(0.0, stats[:, 2]))
        stats[:, 3] = np.log1p(np.maximum(0.0, stats[:, 3]))
        x = torch.from_numpy(stats)

        if not self.edges:
            return Data(
                x=x,
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2), dtype=torch.float),
                edge_weight=torch.zeros((0,), dtype=torch.float),
            )

        ref_time = current_time or max(e[3] for e in self.edges)

        src_list = []
        dst_list = []
        attr_list = []
        weight_list = []

        for src, dst, amount, ts in self.edges:
            src_list.append(src)
            dst_list.append(dst)

            delta_sec = max(0.0, (ref_time - ts).total_seconds())
            decay = float(np.exp(-delta_sec / max(1.0, self.decay_tau_seconds)))

            norm_amt = float(np.log1p(max(0.0, amount)))
            attr_list.append([norm_amt, delta_sec / 3600.0])
            weight_list.append(decay)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float)
        edge_weight = torch.tensor(weight_list, dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
        )

    def get_k_hop_subgraph(
        self,
        entity_id: str,
        num_hops: int = 2,
        current_time: datetime | None = None,
    ) -> tuple[Data, int, list[str]]:
        """
        Extracts an ego-network k-hop subgraph around a given entity.
        Used by GNNExplainer and the Streamlit case inspection UI.

        Args:
            entity_id: Account ID of interest.
            num_hops: Number of neighborhood expansion hops.
            current_time: Reference timestamp.

        Returns:
            Tuple of (subgraph_data, target_node_subgraph_idx, list_of_node_entity_ids).
        """
        if entity_id not in self.node_to_idx:
            raise KeyError(f"Entity '{entity_id}' not found in dynamic graph.")

        target_idx = self.node_to_idx[entity_id]
        full_data = self.build_pyg_data(current_time=current_time)

        # Build undirected edge index so ego-network captures both payers and payees
        undirected_edges = to_undirected(full_data.edge_index, num_nodes=full_data.num_nodes)

        subset, _, mapping, _ = k_hop_subgraph(
            node_idx=target_idx,
            num_hops=num_hops,
            edge_index=undirected_edges,
            relabel_nodes=True,
            num_nodes=full_data.num_nodes,
        )

        # Filter directed edges belonging to induced subgraph
        sub_nodes_set = set(subset.tolist())
        src_mask = torch.tensor([s.item() in sub_nodes_set for s in full_data.edge_index[0]])
        dst_mask = torch.tensor([d.item() in sub_nodes_set for d in full_data.edge_index[1]])
        edge_mask = src_mask & dst_mask

        sub_edge_index = full_data.edge_index[:, edge_mask]
        # Relabel sub_edge_index
        relabel_map = {int(old_id): new_id for new_id, old_id in enumerate(subset.tolist())}
        sub_src = torch.tensor([relabel_map[int(s)] for s in sub_edge_index[0]], dtype=torch.long)
        sub_dst = torch.tensor([relabel_map[int(d)] for d in sub_edge_index[1]], dtype=torch.long)
        relabeled_edge_index = (
            torch.stack([sub_src, sub_dst], dim=0)
            if len(sub_src) > 0
            else torch.zeros((2, 0), dtype=torch.long)
        )

        sub_x = full_data.x[subset]
        sub_edge_attr = full_data.edge_attr[edge_mask] if full_data.edge_attr is not None else None
        sub_edge_weight = (
            full_data.edge_weight[edge_mask] if full_data.edge_weight is not None else None
        )

        sub_data = Data(
            x=sub_x,
            edge_index=relabeled_edge_index,
            edge_attr=sub_edge_attr,
            edge_weight=sub_edge_weight,
            num_nodes=len(subset),
        )
        sub_node_entities = [self.idx_to_node[int(orig_idx)] for orig_idx in subset]

        return sub_data, int(mapping[0]), sub_node_entities
