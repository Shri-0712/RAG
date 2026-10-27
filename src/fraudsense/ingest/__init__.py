"""
Ingestion module for FraudSense.
Normalizes public datasets into a unified time-ordered payment event stream.
"""

from .european_loader import load_european
from .ieee_loader import load_ieee
from .paysim_loader import load_paysim
from .replay import EventReplayEngine
from .schema import UNIFIED_COLUMNS, TransactionEvent

__all__ = [
    "UNIFIED_COLUMNS",
    "EventReplayEngine",
    "TransactionEvent",
    "load_european",
    "load_ieee",
    "load_paysim",
]
