"""
Unit tests for configuration parameters and regulatory policy clauses.
"""

import json

from fraudsense.config import get_project_root, load_config


def test_load_config_structure():
    """Verify params.yaml loads and contains all required sections."""
    cfg = load_config()
    assert isinstance(cfg, dict)

    # Required top-level sections
    expected_sections = [
        "project",
        "data",
        "decision",
        "timers",
        "reward",
        "labels",
        "feedback",
        "simulation",
        "models",
        "monitoring",
        "retraining",
        "enrichment",
    ]
    for section in expected_sections:
        assert section in cfg, f"Missing section '{section}' in params.yaml"

    # Specific non-negotiable parameter values
    assert cfg["decision"]["high_value_threshold"] == 50000
    assert cfg["decision"]["conformal_coverage"] == 0.90
    assert cfg["reward"]["alpha"] == 1.0
    assert cfg["reward"]["gamma"] == 0.9
    assert cfg["labels"]["provisional_labels_in_training"] is False


def test_clauses_json_integrity():
    """Verify data/policy/clauses.json exists and strictly adheres to schema."""
    clauses_path = get_project_root() / "data" / "policy" / "clauses.json"
    assert clauses_path.exists(), "data/policy/clauses.json missing"

    with open(clauses_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "corpus_version" in data
    assert "clauses" in data
    assert len(data["clauses"]) >= 5

    required_clause_keys = ["id", "source", "para", "topic", "summary", "verified"]
    clause_ids = set()
    for clause in data["clauses"]:
        for key in required_clause_keys:
            assert key in clause, f"Clause {clause} missing key '{key}'"
        assert clause["id"].startswith("MD-"), f"Clause ID {clause['id']} does not start with MD-"
        assert clause["id"] not in clause_ids, f"Duplicate clause ID: {clause['id']}"
        clause_ids.add(clause["id"])
