"""
Unit tests for FastAPI sub-200ms transaction scoring API.
"""

from fastapi.testclient import TestClient

from fraudsense.audit.log import DatabaseManager
from fraudsense.scoring.api import create_app


def test_scoring_api_health_check(tmp_path):
    """Verify health endpoint."""
    db = DatabaseManager(db_path=tmp_path / "test_api.db")
    app = create_app(db_manager=db)
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["prototype_mode"] == "SIMULATED"
    assert data["fast_path_llm_calls"] == 0


def test_scoring_api_score_transaction(tmp_path):
    """Verify /score endpoint scores transaction, persists SQLite records, and returns decision."""
    db = DatabaseManager(db_path=tmp_path / "test_api.db")
    app = create_app(db_manager=db)
    client = TestClient(app)

    payload = {
        "event_id": "EVT_API_001",
        "amount": 12500.0,
        "payer_id": "ACC_PAYER_API",
        "payee_id": "ACC_PAYEE_API",
        "channel": "TRANSFER",
        "device_id": "DEV_API_1",
    }

    res = client.post("/score", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["event_id"] == "EVT_API_001"
    assert data["action"] in {"ALLOW", "CHALLENGE", "HOLD_FOR_REVIEW"}
    assert "conformal_set" in data
    assert "latency_ms" in data
    assert data["latency_ms"] < 200.0  # Fast-path latency guarantee
    assert data["simulated_flag"] == "SIMULATED"

    # Verify metrics endpoint updates
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    m_data = metrics_res.json()
    assert m_data["total_scored"] == 1
    assert m_data["latency_budget_ms"] == 200.0
