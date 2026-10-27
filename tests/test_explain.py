"""
Unit tests for SHAP tabular reason codes and GNNExplainer.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from fraudsense.models.explain import RiskHeadExplainer, SubgraphExplainerService
from fraudsense.models.gat import GATEncoder


def test_risk_head_explainer_top5_reasons():
    """Verify SHAP reason code generation outputs top-5 factors with directions."""
    feature_names = [f"feat_{i}" for i in range(18)]
    explainer = RiskHeadExplainer(feature_names=feature_names)

    test_features = np.zeros(18, dtype=np.float32)
    test_features[2] = 50.0  # Big positive outlier
    test_features[7] = -20.0
    test_features[12] = 10.0

    reasons = explainer.explain_instance(test_features, top_k=5)
    assert len(reasons) == 5
    # The largest outlier (feat_2) must be the first reason code
    assert reasons[0]["feature"] == "feat_2"
    assert reasons[0]["direction"] == "INCREASES_RISK"
    assert "impact" in reasons[0]
    assert "value" in reasons[0]


def test_subgraph_explainer_service():
    """Verify GNNExplainer produces edge importance on ego-network."""
    gat = GATEncoder(in_channels=4, hidden_channels=16, out_channels=64, heads=4)
    explainer_service = SubgraphExplainerService(model=gat, epochs=2)

    # 4-node triangle/fan subgraph
    x = torch.randn((4, 4), dtype=torch.float)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    subgraph_data = Data(x=x, edge_index=edge_index, num_nodes=4)

    res = explainer_service.explain_subgraph(subgraph_data, target_node_idx=0)
    assert "target_node" in res
    assert res["target_node"] == 0
    assert "top_edges" in res
