"""
Feedback and labels module for FraudSense.
Simulates label maturity delays, random audits of ALLOW decisions, and importance weighting.
"""

from .label_service import LabelService, LabelSource, TrainingExample

__all__ = ["LabelService", "LabelSource", "TrainingExample"]
