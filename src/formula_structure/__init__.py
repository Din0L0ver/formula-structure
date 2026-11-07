"""Structural diagnostics that narrow the search space of symbolic regression."""

from .analyzer import AnalysisResult, StructureAnalyzer
from .contrasts import FrozenContrastAnalyzer, FrozenContrastResult
from .datasets import SyntheticEngineeringData, make_synthetic_engineering_data
from .modules import (
    FormulaModule,
    IterativeModuleRegressor,
    OptunaPowerFormulaRegressor,
    expanded_graph_modules,
    guided_modules,
    initial_exponents_from_analysis,
    module_graph_from_selection,
)
from .regimes import RegimeDiscoverer, RegimeDiscoveryResult

__all__ = [
    "AnalysisResult",
    "StructureAnalyzer",
    "FrozenContrastAnalyzer",
    "FrozenContrastResult",
    "SyntheticEngineeringData",
    "make_synthetic_engineering_data",
    "FormulaModule",
    "IterativeModuleRegressor",
    "OptunaPowerFormulaRegressor",
    "expanded_graph_modules",
    "guided_modules",
    "initial_exponents_from_analysis",
    "module_graph_from_selection",
    "RegimeDiscoverer",
    "RegimeDiscoveryResult",
]
__version__ = "0.1.0"
