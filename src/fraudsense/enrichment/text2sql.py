"""Guarded, read-only SQLite queries for simulated FraudSense case enrichment."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from ..audit.log import AuditLogger, DatabaseManager
from ..config import load_config


class UnsafeQueryError(ValueError):
    """Raised when a question or SQL statement fails slow-path guardrails."""


@dataclass(frozen=True)
class QueryResult:
    """Auditable result of a guarded, read-only query."""

    sql: str
    rows: list[dict[str, Any]]
    elapsed_ms: float
    simulated_flag: str = "SIMULATED"


class Text2SQLService:
    """Maps a small approved question set to parameterised SQLite SELECT queries."""

    _INJECTION = re.compile(r"ignore\s+(all\s+)?previous|system\s+prompt|jailbreak", re.IGNORECASE)
    _FORBIDDEN = re.compile(
        r"\b(attach|pragma|insert|update|delete|drop|alter|create|replace|vacuum|grant|revoke|"
        r"transaction|begin|commit|rollback)\b|--|/\*|;",
        re.IGNORECASE,
    )
    _TABLE = re.compile(r"\b(?:from|join)\s+([a-z_]+)", re.IGNORECASE)

    def __init__(
        self, db_manager: DatabaseManager | None = None, config: dict[str, Any] | None = None
    ):
        self.config = config or load_config()
        self.db = db_manager or DatabaseManager()
        self.audit = AuditLogger(self.db)
        settings = self.config.get("enrichment", {}).get("text2sql", {})
        self.row_limit = int(settings.get("row_limit", 500))
        self.time_limit_s = float(settings.get("time_limit_s", 5))
        self.allowed_tables = set(settings.get("allowed_tables", []))

    def answer_question(self, question: str, case_id: str) -> QueryResult:
        """Uses deterministic templates for account history, counterparties, and devices."""
        if self._INJECTION.search(question):
            raise UnsafeQueryError("Prompt-injection language is not accepted.")
        case = self._case(case_id)
        normalized = question.lower()
        if any(term in normalized for term in ("history", "payment", "transaction")):
            sql = "SELECT event_id, ts, amount, payer_id, payee_id, channel FROM payment_event WHERE payer_id = :payer_id ORDER BY ts DESC LIMIT :limit"
            params = {"payer_id": case["payer_id"], "limit": self.row_limit}
        elif any(term in normalized for term in ("related", "counterpart", "payee")):
            sql = "SELECT payee_id, COUNT(*) AS payment_count, SUM(amount) AS total_amount FROM payment_event WHERE payer_id = :payer_id GROUP BY payee_id ORDER BY payment_count DESC LIMIT :limit"
            params = {"payer_id": case["payer_id"], "limit": self.row_limit}
        elif any(term in normalized for term in ("device", "shared")):
            sql = "SELECT event_id, ts, amount, payer_id, payee_id FROM payment_event WHERE device_id = :device_id ORDER BY ts DESC LIMIT :limit"
            params = {"device_id": case["device_id"], "limit": self.row_limit}
        else:
            raise UnsafeQueryError("Question is outside the approved case-query templates.")
        return self.execute_select(sql, params, case_id)

    def execute_select(
        self, sql: str, params: dict[str, Any] | None = None, case_id: str | None = None
    ) -> QueryResult:
        """Validates and executes one bounded SELECT statement with bound parameters."""
        self._validate_sql(sql)
        start = time.perf_counter()
        session = self.db.get_session()
        try:
            result = session.execute(text(sql), params or {})
            rows = [dict(row) for row in result.mappings().fetchmany(self.row_limit)]
        finally:
            session.close()
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > self.time_limit_s * 1000:
            raise UnsafeQueryError("Query exceeded the configured execution-time limit.")
        if case_id:
            self.audit.log(
                "CASE",
                case_id,
                "TEXT2SQL_QUERY",
                {"sql": sql, "row_count": len(rows), "simulated_flag": "SIMULATED"},
            )
        return QueryResult(sql=sql, rows=rows, elapsed_ms=elapsed_ms)

    def _validate_sql(self, sql: str) -> None:
        statement = sql.strip()
        if not statement.lower().startswith("select") or self._FORBIDDEN.search(statement):
            raise UnsafeQueryError("Only one read-only SELECT statement is allowed.")
        tables = {match.group(1).lower() for match in self._TABLE.finditer(statement)}
        if not tables or not tables.issubset(self.allowed_tables):
            raise UnsafeQueryError("SQL references a table outside the allowlist.")
        limit_match = re.search(r"\blimit\s+(\d+)\b", statement, re.IGNORECASE)
        if limit_match and int(limit_match.group(1)) > self.row_limit:
            raise UnsafeQueryError("SQL exceeds the configured row limit.")

    def _case(self, case_id: str) -> dict[str, str]:
        session = self.db.get_session()
        try:
            row = (
                session.execute(
                    text(
                        "SELECT p.payer_id, p.device_id FROM case_record c JOIN payment_event p ON c.event_id = p.event_id WHERE c.case_id = :case_id"
                    ),
                    {"case_id": case_id},
                )
                .mappings()
                .one_or_none()
            )
        finally:
            session.close()
        if row is None:
            raise ValueError(f"Unknown case_id: {case_id}")
        return dict(row)
