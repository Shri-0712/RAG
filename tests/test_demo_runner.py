"""End-to-end tests for the explicitly simulated dashboard scenarios."""

from fraudsense.audit.log import DatabaseManager, RewardRecord
from fraudsense.simulation.demo_runner import DemoScenarioRunner


def test_high_value_fraud_scenario_opens_case_and_writes_reward(tmp_path):
    db = DatabaseManager(tmp_path / "demo.db")
    result = DemoScenarioRunner(db).run("High-value mule risk")
    assert result["final_action"] == "HOLD_FOR_REVIEW"
    assert result["case_id"] is not None
    assert all(stage["status"] == "PASS" for stage in result["stages"])
    session = db.get_session()
    try:
        assert session.query(RewardRecord).filter_by(decision_id=result["decision_id"]).count() == 1
    finally:
        session.close()


def test_timeout_fixture_never_releases_funds(tmp_path):
    result = DemoScenarioRunner(DatabaseManager(tmp_path / "demo.db")).run(
        "Challenge timeout containment"
    )
    assert result["challenge_status"] == "TIMED_OUT"
    assert result["final_action"] == "HOLD_FOR_REVIEW"
