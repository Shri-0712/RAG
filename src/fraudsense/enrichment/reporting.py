"""Local-only drafts of simulated CPFIR and FMR reporting checklists."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ..audit.log import AuditLogger, DatabaseManager, ReportingRecord
from ..config import get_project_root


class ReportingService:
    """Creates labelled local drafts; it never submits any report externally."""

    def __init__(self, db_manager: DatabaseManager | None = None, output_dir: Path | None = None):
        self.db = db_manager or DatabaseManager()
        self.audit = AuditLogger(self.db)
        self.output_dir = output_dir or get_project_root() / "reports" / "simulated_drafts"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def draft(self, case_id: str, report_type: str, clause_id: str) -> Path:
        """Writes a checklist draft for CPFIR or FMR only, marked SIMULATED."""
        report_type = report_type.upper()
        if report_type not in {"CPFIR", "FMR"}:
            raise ValueError("report_type must be CPFIR or FMR")
        payload = {
            "status": "SIMULATED — NOT FOR SUBMISSION",
            "case_id": case_id,
            "report_type": report_type,
            "clause_id": clause_id,
            "created_at": datetime.now(UTC).isoformat(),
            "checklist": [
                "Confirm simulated case facts",
                "Obtain human compliance review",
                "Do not submit externally",
            ],
        }
        path = self.output_dir / f"SIMULATED_{report_type}_{case_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        session = self.db.get_session()
        try:
            session.add(
                ReportingRecord(
                    report_id=f"SIM_{report_type}_{case_id}",
                    case_id=case_id,
                    report_type=report_type,
                    draft_json=json.dumps(payload),
                    simulated_flag="SIMULATED",
                )
            )
            session.commit()
        finally:
            session.close()
        self.audit.log(
            "CASE",
            case_id,
            "REPORT_DRAFTED",
            {"report_type": report_type, "path": str(path), "simulated_flag": "SIMULATED"},
        )
        return path
