"""
Slow-path enrichment module for FraudSense.
Handles guarded Text2SQL queries, CRAG-style RAG over RBI clauses, and draft regulatory reports.
"""

from .rag import PolicyRAG
from .reporting import ReportingService
from .text2sql import Text2SQLService, UnsafeQueryError

__all__ = ["PolicyRAG", "ReportingService", "Text2SQLService", "UnsafeQueryError"]
