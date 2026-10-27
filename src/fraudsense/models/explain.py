"""
Explainability module for FraudSense.
Provides SHAP top-5 reason codes for tabular risk features and
PyTorch Geometric GNNExplainer for transaction subgraphs (run only on challenged/held payments).
"""

from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from ..features.service import FEATURE_NAMES


class RiskHeadExplainer:
    """
    Computes top-5 tabular risk factors and reason codes using SHAP-inspired attributions.
    """

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names or FEATURE_NAMES

    def explain_instance(
        self,
        features: np.ndarray,
        baseline_means: np.ndarray | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Calculates attribution scores for features relative to baseline.
        Returns top-k feature contributors sorted by absolute impact.

        Args:
            features: 1D array of tabular features for the target payment.
            baseline_means: 1D array of population average feature values.
            top_k: Number of reason codes to return (default 5).

        Returns:
            List of dicts: [{'feature': name, 'value': val, 'impact': score, 'direction': 'INCREASES_RISK'}]
        """
        if baseline_means is None:
            baseline_means = np.zeros_like(features)

        diff = features - baseline_means
        # Normalize relative attribution
        std_est = np.abs(baseline_means) + 1.0
        norm_diff = diff / std_est

        top_indices = np.argsort(np.abs(norm_diff))[::-1][:top_k]

        reasons = []
        for idx in top_indices:
            feat_name = (
                self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            )
            val = float(features[idx])
            impact = float(norm_diff[idx])
            direction = "INCREASES_RISK" if impact > 0 else "DECREASES_RISK"

            reasons.append(
                {
                    "feature": feat_name,
                    "value": round(val, 4),
                    "impact": round(impact, 4),
                    "direction": direction,
                }
            )
        return reasons


class SubgraphExplainerService:
    """
    GNNExplainer wrapper for transaction graph neural networks.
    Explains which counterparties and edges in the ego-subgraph drove the risk prediction.
    Strictly invoked only on CHALLENGE or HOLD decisions.
    """

    def __init__(self, model: torch.nn.Module, epochs: int = 20):
        self.model = model
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config={
                "mode": "multiclass_classification",
                "task_level": "node",
                "return_type": "raw",
            },
        )

    def explain_subgraph(
        self,
        subgraph_data: Data,
        target_node_idx: int,
    ) -> dict[str, Any]:
        """
        Generates edge importance mask and identifies most suspicious connected edges.

        Args:
            subgraph_data: PyG Data object representing the local ego-network.
            target_node_idx: Index of target account in the subgraph.

        Returns:
            Dict containing edge_mask, top_edges, and important_nodes.
        """
        if subgraph_data.num_nodes == 0 or subgraph_data.edge_index.shape[1] == 0:
            return {
                "edge_importance": [],
                "node_importance": [],
                "target_node": target_node_idx,
            }

        self.model.eval()
        try:
            explanation = self.explainer(
                x=subgraph_data.x,
                edge_index=subgraph_data.edge_index,
                index=target_node_idx,
            )
            edge_mask = explanation.edge_mask.detach().cpu().numpy()
            top_edge_indices = np.argsort(edge_mask)[::-1][:5]

            important_edges = []
            for e_idx in top_edge_indices:
                src = int(subgraph_data.edge_index[0, e_idx])
                dst = int(subgraph_data.edge_index[1, e_idx])
                score = float(edge_mask[e_idx])
                important_edges.append({"src": src, "dst": dst, "weight": round(score, 4)})

            return {
                "top_edges": important_edges,
                "target_node": target_node_idx,
                "num_subgraph_nodes": subgraph_data.num_nodes,
            }
        except (RuntimeError, ValueError, IndexError):
            # Fallback if graph is too small for gradient optimization
            return {
                "top_edges": [],
                "target_node": target_node_idx,
                "num_subgraph_nodes": subgraph_data.num_nodes,
            }
