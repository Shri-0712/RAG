"""
Simulation module for FraudSense.
Simulates customer responses, analyst decisions, and virtual replay clock progression.
"""
"""Simulated customers, analysts, and repeatable demo scenarios."""

from .demo_runner import SCENARIOS, DemoScenarioRunner

__all__ = ["SCENARIOS", "DemoScenarioRunner"]
