"""Option Collar Framework - Core application module."""

from src.builder import CollarBuilder
from src.capped_gains import CappedGainsAnalyzer, CappedGainsResult, UpsideScenario
from src.comparator import CollarComparator
from src.greeks import GreeksAnalyzer
from src.scenario import NAMED_SCENARIOS, Scenario, ScenarioAnalyzer, ScenarioResult
from src.structures import CollarLeg, CollarPosition, CollarPricing
from src.visualizer import CollarVisualizer

__all__ = [
    "CappedGainsAnalyzer",
    "CappedGainsResult",
    "CollarBuilder",
    "CollarComparator",
    "CollarLeg",
    "CollarPosition",
    "CollarPricing",
    "CollarVisualizer",
    "GreeksAnalyzer",
    "NAMED_SCENARIOS",
    "Scenario",
    "ScenarioAnalyzer",
    "ScenarioResult",
    "UpsideScenario",
]
