"""Closed-world, deterministic policy retrieval over the supplied clause corpus."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ..config import get_project_root, load_config


@dataclass(frozen=True)
class RetrievedClause:
    """One retrieved, corpus-backed policy clause."""

    clause_id: str
    topic: str
    summary: str
    score: float


class PolicyRAG:
    """BM25 plus token-overlap retrieval with citation-only deterministic answers."""

    def __init__(self, corpus_path: Path | None = None, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        settings = cfg.get("enrichment", {}).get("rag", {})
        self.top_k = int(settings.get("top_k", 3))
        self.minimum_score = float(settings.get("min_retrieval_score", 0.35))
        path = corpus_path or get_project_root() / "data" / "policy" / "clauses.json"
        self.clauses = json.loads(path.read_text(encoding="utf-8"))["clauses"]
        self.documents = [
            self._tokens(f"{c['id']} {c['topic']} {c['summary']}") for c in self.clauses
        ]
        self.bm25 = BM25Okapi(self.documents)
        self._injection = re.compile(
            r"ignore\s+(all\s+)?previous|system\s+prompt|jailbreak|invent\s+(a\s+)?(new\s+)?clause",
            re.IGNORECASE,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedClause]:
        """Returns only verified corpus clauses, ranked by fused lexical evidence."""
        if self._injection.search(question):
            return []
        query = self._tokens(question)
        if not query:
            return []
        bm25_scores = self.bm25.get_scores(query)
        max_score = max(bm25_scores) if len(bm25_scores) else 0.0
        ranked: list[RetrievedClause] = []
        for index, clause in enumerate(self.clauses):
            if not clause.get("verified", False):
                continue
            overlap = len(set(query) & set(self.documents[index])) / max(1, len(set(query)))
            lexical = float(bm25_scores[index] / max_score) if max_score else 0.0
            score = 0.7 * lexical + 0.3 * overlap
            ranked.append(RetrievedClause(clause["id"], clause["topic"], clause["summary"], score))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[: top_k or self.top_k]

    def answer(self, question: str) -> str:
        """Returns a grounded response or the prescribed low-confidence fallback."""
        clauses = self.retrieve(question)
        if not clauses or clauses[0].score < self.minimum_score:
            return "No applicable clause found"
        return " ".join(f"[{c.clause_id}] {c.summary}" for c in clauses)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
