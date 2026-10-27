"""
Unit tests for dynamic graph builder and recency-weighted decay.
"""

from datetime import UTC, datetime, timedelta

from torch_geometric.data import Data

from fraudsense.graph.builder import DynamicGraphBuilder
from fraudsense.ingest.schema import TransactionEvent


def test_dynamic_graph_builder_ingestion():
    """Verify adding events creates nodes, edges, and valid PyG Data object."""
    builder = DynamicGraphBuilder()
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

    e1 = TransactionEvent(
        event_id="E1",
        ts=base_time,
        amount=1000.0,
        payer_id="ACC_A",
        payee_id="ACC_B",
        channel="TRANSFER",
        device_id="DEV_1",
        label=0,
    )
    e2 = TransactionEvent(
        event_id="E2",
        ts=base_time + timedelta(hours=2),
        amount=5000.0,
        payer_id="ACC_B",
        payee_id="ACC_C",
        channel="TRANSFER",
        device_id="DEV_2",
        label=1,
    )

    builder.add_event(e1)
    builder.add_event(e2)

    assert len(builder.node_to_idx) == 3
    assert "ACC_A" in builder.node_to_idx
    assert "ACC_B" in builder.node_to_idx
    assert "ACC_C" in builder.node_to_idx

    pyg_data = builder.build_pyg_data()
    assert isinstance(pyg_data, Data)
    assert pyg_data.num_nodes == 3
    assert pyg_data.edge_index.shape == (2, 2)
    assert pyg_data.edge_weight.shape == (2,)
    assert pyg_data.x.shape == (3, 4)


def test_dynamic_graph_recency_decay():
    """Verify older edges receive strictly lower weights due to exponential decay."""
    builder = DynamicGraphBuilder()
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

    # Edge 1 is 100 hours old
    e1 = TransactionEvent(
        event_id="E1",
        ts=base_time,
        amount=100.0,
        payer_id="ACC_X",
        payee_id="ACC_Y",
        channel="TRANSFER",
        device_id="DEV_X",
        label=0,
    )
    # Edge 2 is 1 hour old
    e2 = TransactionEvent(
        event_id="E2",
        ts=base_time + timedelta(hours=99),
        amount=100.0,
        payer_id="ACC_Y",
        payee_id="ACC_Z",
        channel="TRANSFER",
        device_id="DEV_Y",
        label=0,
    )

    builder.add_event(e1)
    builder.add_event(e2)

    current_time = base_time + timedelta(hours=100)
    data = builder.build_pyg_data(current_time=current_time)

    w_old = data.edge_weight[0].item()
    w_recent = data.edge_weight[1].item()

    assert w_recent > w_old
    assert 0.0 < w_old < 1.0
    assert 0.0 < w_recent <= 1.0


def test_get_k_hop_subgraph():
    """Verify extraction of local ego-network around target node."""
    builder = DynamicGraphBuilder()
    base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

    # Chain: A -> B -> C -> D
    nodes = ["N_A", "N_B", "N_C", "N_D"]
    for i in range(len(nodes) - 1):
        builder.add_event(
            TransactionEvent(
                event_id=f"E_{i}",
                ts=base_time + timedelta(minutes=i * 10),
                amount=500.0,
                payer_id=nodes[i],
                payee_id=nodes[i + 1],
                channel="TRANSFER",
                device_id=f"D_{i}",
                label=0,
            )
        )

    # 1-hop subgraph around N_B should include N_A, N_B, N_C but not N_D
    _sub_data, target_idx, sub_entities = builder.get_k_hop_subgraph("N_B", num_hops=1)
    assert "N_B" in sub_entities
    assert sub_entities[target_idx] == "N_B"
    assert "N_A" in sub_entities
    assert "N_C" in sub_entities
    assert "N_D" not in sub_entities
